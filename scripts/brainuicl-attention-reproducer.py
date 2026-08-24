#!/usr/bin/env python3
"""Minimal deterministic reproducer for BrainUICL's custom attention path.

This is intentionally independent of the BrainUICL source tree: it mirrors the
Q/K/V -> scaled matmul -> softmax(dim=1) -> context -> dense sequence with
fixed weights and input shape [1, 20, 512]. It is a compiler diagnostic, not a
model-quality or deployment benchmark.
"""

from __future__ import annotations

import argparse
import json
import platform
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".edgeforge/attention-reproducer-v11.json")
    args = parser.parse_args()
    import torch

    torch.manual_seed(4321)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class BrainUICLAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w_query = torch.nn.Linear(512, 512)
            self.w_key = torch.nn.Linear(512, 512)
            self.w_value = torch.nn.Linear(512, 512)
            self.dense = torch.nn.Linear(512, 512)

        def forward(self, x):
            batch = x.shape[0]
            query = self.w_query(x).view(batch, -1, 8, 64).permute(0, 2, 1, 3)
            key = self.w_key(x).view(batch, -1, 8, 64).permute(0, 2, 1, 3)
            value = self.w_value(x).view(batch, -1, 8, 64).permute(0, 2, 1, 3)
            score = torch.matmul(query, key.transpose(-1, -2)) / (x.shape[2] ** 0.5)
            probability = torch.softmax(score, dim=1)
            context = torch.matmul(probability, value)
            context = context.permute(0, 2, 1, 3).contiguous().view(batch, -1, 512)
            return self.dense(context)

    model = BrainUICLAttention().eval().to(device)
    sample = torch.randn(1, 20, 512, device=device)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        eager = model(sample)
        outputs = {"eager": eager}
        timings = {}
        for name, candidate in (
            ("inductor", torch.compile(model, backend="inductor", mode="reduce-overhead")),
            ("inductor-no-pattern", torch.compile(model, backend="inductor", options={"pattern_matcher": False})),
        ):
            sync()
            started = time.perf_counter()
            output = candidate(sample)
            sync()
            timings[name] = (time.perf_counter() - started) * 1000.0
            outputs[name] = output.detach()

    result = {
        "schema_version": 1,
        "input_shape": [1, 20, 512],
        "num_heads": 8,
        "softmax_dim": 1,
        "seed": 4321,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "first_call_ms": timings,
        "comparisons": {
            name: {
                "max_abs_error": float((eager - value).abs().max().item()),
                "mean_abs_error": float((eager - value).abs().mean().item()),
                "allclose": bool(torch.allclose(eager, value, rtol=2e-3, atol=2e-4)),
            }
            for name, value in outputs.items()
            if name != "eager"
        },
    }
    output_path = __import__("pathlib").Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
