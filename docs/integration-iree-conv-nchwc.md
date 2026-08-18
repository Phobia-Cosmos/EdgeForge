# EdgeForge × IREE `conv_nchwc` dilation

This integration keeps IREE as a separate source repository and exposes the existing IREE `#24760` CPU ukernel work through EdgeForge's Kernel Registry and `compile → correctness → benchmark` contract. EdgeForge does not copy LLVM, the IREE source tree, or model/data payloads into its repository.

## Current implementation

- EdgeForge Operator IR now accepts `conv_nchwc` with packed shape `[N, OC_outer, OH, OW, IC_outer, FH, FW, k0, c0]` and `stride_h/stride_w/dilation_h/dilation_w` attributes.
- The `iree-ukernel` backend executes explicitly registered prebuilt IREE test and benchmark binaries with `shell=False`, worker work-root containment, and the worker executable allow-list.
- The backend records the IREE repository, commit and patch digest in the compiler manifest Artifact. The compile stage is intentionally marked `prebuilt`; this is runtime execution evidence, not a claim that the IREE compiler was built.
- The benchmark adapter forwards packed shape, stride, dilation and accumulate attributes to the IREE benchmark flags and parses real-time and items-per-second counters.

## Kernel registration payload

Register this as a trusted local kernel only after checking the two binary paths on the target worker. Paths are relative to the worker's `--work-root`:

```json
{
  "id": "kernel-iree-conv-nchwc-dilation-arm64-v1",
  "operator": "conv_nchwc",
  "backend": "iree-ukernel",
  "version": "issue-24760-dilated-conv",
  "architectures": ["aarch64"],
  "dtypes": ["fp32"],
  "shape_constraints": {"k0": 16, "c0": 16},
  "metadata": {
    "trusted": true,
    "test_command": ["iree-build-runtime/runtime/src/iree/builtins/ukernel/tools/conv_nchwc_test"],
    "benchmark_command": ["iree-build-runtime/runtime/src/iree/builtins/ukernel/tools/conv_nchwc_benchmark", "--benchmark_min_time=0s"],
    "workdir": "iree-build-runtime",
    "k0": 16,
    "c0": 16,
    "compiler": "iree-runtime",
    "iree_repository": "https://github.com/iree-org/iree.git",
    "iree_commit": "a82330a3343d84211e71d2596725c8185227b29f",
    "patch_sha256": "22ff9075b947e9b232b546381b899d35e50e02b794e1ac7f651c06a54681ac33",
    "validation_status": "pending-orangepi"
  },
  "compiler": {"name": "iree", "mode": "runtime-only-prebuilt"}
}
```

Use `config/iree-conv-nchwc-dilation-arm64-v1.json` for Orange Pi and `config/iree-conv-nchwc-dilation-x86-v1.json` for the current host validation. They deliberately use different Kernel IDs so architecture-specific Artifact and Benchmark evidence cannot overwrite each other. The control plane treats registered kernel metadata as trusted configuration. Keep registration restricted to the project operator, and pass both binary paths explicitly when starting a worker:

```bash
PYTHONPATH=src python3 - <<'PY'
import json, os
from pathlib import Path
from edgeforge.client import Client

spec = json.loads(Path("config/iree-conv-nchwc-dilation-arm64-v1.json").read_text())
print(Client(os.environ["EDGEFORGE_CONTROL_URL"], os.environ["EDGEFORGE_TOKEN"]).request("POST", "/api/v1/kernels", spec))
PY
```

```bash
export EDGEFORGE_WORK_ROOT=/home/undefined/UbuntuData/Projects
export IREE_TEST=/home/undefined/UbuntuData/Projects/iree-build-runtime/runtime/src/iree/builtins/ukernel/tools/conv_nchwc_test
export IREE_BENCH=/home/undefined/UbuntuData/Projects/iree-build-runtime/runtime/src/iree/builtins/ukernel/tools/conv_nchwc_benchmark

PYTHONPATH=src python3 -m edgeforge worker \
  --control-url "$EDGEFORGE_CONTROL_URL" \
  --token "$EDGEFORGE_TOKEN" \
  --worker-id worker-4070s \
  --work-root "$EDGEFORGE_WORK_ROOT" \
  --allow-command "$IREE_TEST" \
  --allow-command "$IREE_BENCH" \
  --label backend=iree-ukernel
```

For Orange Pi, use the same relative metadata and set the worker root to the directory containing the ARM64 build. The first board run should use the generic fallback binary and must not be labelled as x86 AVX-512 evidence.

## Running a dilation pipeline

The packed shape below matches the asymmetric IREE test case and exercises `dilation_h=2,dilation_w=3`:

```bash
PYTHONPATH=src python3 -m edgeforge kernel-pipeline \
  --control-url "$EDGEFORGE_CONTROL_URL" \
  --token "$EDGEFORGE_TOKEN" \
  --kernel-id kernel-iree-conv-nchwc-dilation-arm64-v1 \
  --operator conv_nchwc \
  --shape 1,1,3,5,1,3,3,16,16 \
  --dtype fp32 \
  --attrs '{"stride_h":1,"stride_w":1,"dilation_h":2,"dilation_w":3}' \
  --worker-id worker-orangepi \
  --repeats 3 \
  --wait
```

The returned result has the standard EdgeForge pipeline stages. `correctness` comes from `conv_nchwc_test`; benchmark timings and throughput are parsed from `conv_nchwc_benchmark`; the uploaded compiler manifest binds the IREE commit and patch identity to the resulting Artifact and Benchmark record.

## What this does not claim

- The current pipeline validates prebuilt runtime/ukernel binaries; it does not build the IREE compiler and does not execute the compiler MLIR lit test.
- The current x86 result is local correctness/performance evidence only. Orange Pi evidence requires an ARM64-native or correctly cross-compiled binary and a fresh worker capability snapshot.
- EdgeForge's RA-EEG Experiment Contract remains separate from the operator benchmark. EEG accuracy, plasticity, forgetting and spectrum metrics continue to flow through `experiment_run`/`ExperimentBundle`; `conv_nchwc` records compiler/runtime metrics only.
