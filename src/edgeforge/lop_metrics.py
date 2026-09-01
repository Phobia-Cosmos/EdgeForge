"""PyTorch-optional, architecture-agnostic LoP diagnostics.

The control plane remains dependency-free: PyTorch is imported only inside
the numerical helpers.  A representation is always accompanied by an
explicit feature axis so that ``[B, T, D]`` EEG tokens and ``[B, C, L]``
convolutional feature maps are not silently flattened in different ways.

The functions in this module are descriptive diagnostics.  They do not turn
an effective-rank change into a causal LoP claim.  The primary LoP outcome is
the fixed-budget fresh-vs-warm gap returned by ``fixed_budget_probe``.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


EPS = 1e-12


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised without torch
        raise RuntimeError("PyTorch is required for LoP numerical diagnostics") from error
    return torch


def _as_feature_matrix(
    values: Any,
    *,
    feature_axis: int = -1,
    center: bool = True,
    max_observations: int = 0,
) -> Any:
    """Return a detached ``[observations, features]`` tensor.

    ``feature_axis`` is deliberately required at the adapter boundary.  For
    an EEG tensor ``[B, T, D]`` use ``2``; for a Conv1d map ``[B, C, L]`` use
    ``1`` so that time positions become observations and channels remain
    features.  A deterministic evenly spaced subsample bounds SVD cost.
    """

    torch = _torch()
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    tensor = values.detach().float()
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
        feature_axis = 1
    else:
        axis = int(feature_axis)
        if axis < 0:
            axis += tensor.ndim
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError(f"feature_axis {feature_axis} is invalid for shape {tuple(tensor.shape)}")
        permutation = [index for index in range(tensor.ndim) if index != axis] + [axis]
        tensor = tensor.permute(*permutation).reshape(-1, tensor.shape[axis])
    if max_observations and max_observations > 0 and tensor.shape[0] > max_observations:
        # linspace avoids depending on a random generator and preserves the
        # beginning, middle and end of a sequence/feature-map stream.
        indices = torch.linspace(0, tensor.shape[0] - 1, max_observations, device=tensor.device).round().long()
        tensor = tensor.index_select(0, indices)
    if center and tensor.shape[0] > 1:
        tensor = tensor - tensor.mean(dim=0, keepdim=True)
    return tensor


def spectral_summary(
    values: Any,
    *,
    feature_axis: int = -1,
    center: bool = True,
    max_observations: int = 0,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Compute scale-aware spectrum diagnostics for a representation.

    ``effective_rank`` uses the entropy of singular values, while
    ``stable_rank`` uses squared singular values.  The normalized versions
    divide by the algebraic rank ceiling ``min(N, D)`` and are preferable
    when calibration files or token counts differ between checkpoints.
    """

    torch = _torch()
    matrix = _as_feature_matrix(
        values,
        feature_axis=feature_axis,
        center=center,
        max_observations=max_observations,
    )
    observations, features = (int(matrix.shape[0]), int(matrix.shape[1]))
    singular = torch.linalg.svdvals(matrix)
    rank_ceiling = min(observations, features)
    base: dict[str, Any] = {
        "status": "computed",
        "shape": [int(item) for item in values.shape],
        "feature_axis": int(feature_axis),
        "observation_count": observations,
        "feature_dim": features,
        "rank_ceiling": rank_ceiling,
        "centered": bool(center),
        "max_observations": int(max_observations),
    }
    if singular.numel() == 0:
        return {
            **base,
            "effective_rank": 0.0,
            "effective_rank_normalized": 0.0,
            "stable_rank": 0.0,
            "stable_rank_normalized": 0.0,
            "spectral_entropy": 0.0,
            "spectral_entropy_normalized": 0.0,
            "rank90": 0,
            "rank95": 0,
            "rank99": 0,
            "epsilon_rank": 0,
            "sigma_max": 0.0,
            "sigma_min_positive": None,
            "condition_number": None,
            "tail_energy": 0.0,
            "frobenius_norm": 0.0,
        }
    sigma_max = singular[0]
    total = singular.sum()
    squared = singular.square()
    energy_total = squared.sum()
    if float(total.item()) <= EPS or float(energy_total.item()) <= EPS:
        return {
            **base,
            "effective_rank": 0.0,
            "effective_rank_normalized": 0.0,
            "stable_rank": 0.0,
            "stable_rank_normalized": 0.0,
            "spectral_entropy": 0.0,
            "spectral_entropy_normalized": 0.0,
            "rank90": 0,
            "rank95": 0,
            "rank99": 0,
            "epsilon_rank": 0,
            "sigma_max": 0.0,
            "sigma_min_positive": None,
            "condition_number": None,
            "tail_energy": 0.0,
            "frobenius_norm": float(torch.linalg.vector_norm(matrix).item()),
        }

    probability = singular / total
    entropy = -(probability * probability.clamp_min(EPS).log()).sum()
    cumulative = torch.cumsum(squared / energy_total, dim=0)
    threshold = max(float(epsilon), EPS) * float(sigma_max.item())
    positive = singular[singular > threshold]
    epsilon_rank = int(positive.numel())
    sigma_min_positive = float(positive[-1].item()) if positive.numel() else None
    condition = float(sigma_max.item() / sigma_min_positive) if sigma_min_positive else None

    def energy_rank(target: float) -> int:
        return int(torch.searchsorted(cumulative, torch.tensor(target, device=cumulative.device)).item() + 1)

    max_entropy = math.log(max(1, int(singular.numel())))
    effective = float(entropy.exp().item())
    stable = float((energy_total / sigma_max.square().clamp_min(EPS)).item())
    return {
        **base,
        "effective_rank": effective,
        "effective_rank_normalized": effective / max(1, rank_ceiling),
        "stable_rank": stable,
        "stable_rank_normalized": stable / max(1, rank_ceiling),
        "spectral_entropy": float(entropy.item()),
        "spectral_entropy_normalized": float((entropy / max(max_entropy, EPS)).item()),
        "rank90": energy_rank(0.90),
        "rank95": energy_rank(0.95),
        "rank99": energy_rank(0.99),
        "epsilon_rank": epsilon_rank,
        "sigma_max": float(sigma_max.item()),
        "sigma_min_positive": sigma_min_positive,
        "condition_number": condition,
        "tail_energy": float((1.0 - cumulative[0]).clamp_min(0.0).item()),
        "frobenius_norm": float(torch.linalg.vector_norm(matrix).item()),
    }


def activation_summary(values: Any, *, kind: str = "unknown", zero_threshold: float = 1e-6) -> dict[str, Any]:
    """Summarize dead/near-zero/saturated activation evidence."""

    torch = _torch()
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"status": "empty", "kind": kind, "element_count": 0}
    mean = float(flat.mean().item())
    abs_mean = float(flat.abs().mean().item())
    std = float(flat.std(unbiased=False).item())
    near_zero = float((flat.abs() <= zero_threshold).float().mean().item())
    result: dict[str, Any] = {
        "status": "computed",
        "kind": kind,
        "element_count": int(flat.numel()),
        "mean": mean,
        "mean_abs": abs_mean,
        "second_moment": float(flat.square().mean().item()),
        "std": std,
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "near_zero_fraction": near_zero,
        "dead_fraction": near_zero if kind.lower() == "relu" else None,
        "saturated_fraction": None,
        "zero_threshold": float(zero_threshold),
    }
    lowered = kind.lower()
    if lowered in {"sigmoid", "tanh"}:
        bound = 0.99 if lowered == "sigmoid" else 0.99
        result["saturated_fraction"] = float((flat.abs() >= bound).float().mean().item())
    return result


def linear_cka(left: Any, right: Any, *, feature_axis: int = -1, max_observations: int = 0) -> float | None:
    """Return centered linear CKA, invariant to orthogonal feature rotation."""

    torch = _torch()
    x = _as_feature_matrix(left, feature_axis=feature_axis, center=True, max_observations=max_observations)
    y = _as_feature_matrix(right, feature_axis=feature_axis, center=True, max_observations=max_observations)
    count = min(x.shape[0], y.shape[0])
    if count < 2:
        return None
    x, y = x[:count], y[:count]
    cross = x.T @ y
    numerator = cross.square().sum()
    denominator = torch.sqrt((x.T @ x).square().sum() * (y.T @ y).square().sum()).clamp_min(EPS)
    return float((numerator / denominator).item())


def procrustes_residual(left: Any, right: Any, *, feature_axis: int = -1, max_observations: int = 0) -> float | None:
    """Return relative residual after the best orthogonal feature alignment."""

    torch = _torch()
    x = _as_feature_matrix(left, feature_axis=feature_axis, center=True, max_observations=max_observations)
    y = _as_feature_matrix(right, feature_axis=feature_axis, center=True, max_observations=max_observations)
    count = min(x.shape[0], y.shape[0])
    width = min(x.shape[1], y.shape[1])
    if count < 2 or width < 1:
        return None
    x, y = x[:count, :width], y[:count, :width]
    u, _s, vh = torch.linalg.svd(x.T @ y, full_matrices=False)
    rotation = u @ vh
    residual = torch.linalg.vector_norm(x @ rotation - y)
    return float((residual / torch.linalg.vector_norm(y).clamp_min(EPS)).item())


def attention_summary(attention: Any, *, normalization_axis: int = -1) -> dict[str, Any]:
    """Summarize attention probabilities with the axis recorded explicitly.

    Expected shape is ``[B, H, T, T]`` or ``[B, T, T]``.  The function does
    not silently renormalize across another axis; callers should pass the
    axis used by their implementation and record it in the result.
    """

    torch = _torch()
    if not isinstance(attention, torch.Tensor):
        raise TypeError("attention must be a torch.Tensor")
    values = attention.detach().float()
    axis = int(normalization_axis)
    if axis < 0:
        axis += values.ndim
    if axis < 0 or axis >= values.ndim:
        raise ValueError("normalization_axis is outside attention tensor")
    probabilities = values.clamp_min(EPS)
    entropy = -(probabilities * probabilities.log()).sum(dim=axis)
    axis_length = int(values.shape[axis])
    normalized = entropy / math.log(max(2, axis_length))
    result: dict[str, Any] = {
        "status": "computed",
        "shape": [int(item) for item in values.shape],
        "normalization_axis": axis,
        "normalization_length": axis_length,
        "entropy_mean": float(entropy.mean().item()),
        "entropy_normalized_mean": float(normalized.mean().item()),
        "max_probability": float(values.max().item()),
    }
    if values.ndim >= 3 and values.shape[-1] == values.shape[-2]:
        diagonal = values.diagonal(dim1=-2, dim2=-1)
        result["diagonal_mass"] = float(diagonal.mean().item())
        result["offdiagonal_mass"] = float((1.0 - diagonal.mean()).item())
    if values.ndim == 4:
        # Head diversity is one minus mean pairwise cosine similarity between
        # flattened head maps.  One head has no pairwise comparison.
        heads = values.permute(1, 0, 2, 3).reshape(values.shape[1], -1)
        if heads.shape[0] > 1:
            normalized_heads = heads / torch.linalg.vector_norm(heads, dim=1, keepdim=True).clamp_min(EPS)
            cosine = normalized_heads @ normalized_heads.T
            offdiag = cosine[~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)]
            result["head_diversity"] = float((1.0 - offdiag.mean()).item())
        else:
            result["head_diversity"] = None
    return result


def jacobian_summary(jacobian: Any, *, residual: Any | None = None) -> dict[str, Any]:
    """Summarize a sampled parameter Jacobian and its empirical NTK spectrum."""

    torch = _torch()
    if not isinstance(jacobian, torch.Tensor) or jacobian.ndim != 2:
        raise ValueError("jacobian must be a rank-2 [samples, parameters] tensor")
    rows, parameters = map(int, jacobian.shape)
    result: dict[str, Any] = {
        "status": "computed",
        "scope": "sampled-parameter-jacobian",
        "output_selector": "per-sample mean of all model outputs",
        "sample_count": rows,
        "parameter_dim": parameters,
        "data_dependent": True,
        "jacobian": spectral_summary(jacobian, feature_axis=1, center=True),
    }
    result["jacobian"]["data_dependent"] = True
    kernel = jacobian @ jacobian.T
    kernel_eigenvalues = torch.linalg.eigvalsh((kernel + kernel.T) / 2.0).clamp_min(0.0)
    if kernel_eigenvalues.numel():
        # ``K`` itself is the Gram/kernel matrix.  Taking the SVD of the
        # matrix (rather than of a column containing its eigenvalues) preserves
        # the complete NTK spectrum; the latter would always have rank one and
        # silently turn effective-rank diagnostics into a meaningless value.
        result["ntk"] = spectral_summary(kernel, feature_axis=-1, center=False)
        # Keep dependency provenance at the nested metric level as well as on
        # the enclosing Jacobian record; downstream flatteners often consume
        # ``jacobian.ntk`` directly.
        result["ntk"]["data_dependent"] = True
        result["ntk_eigenvalues"] = kernel_eigenvalues.flip(0).tolist()
        result["ntk_trace"] = float(kernel.trace().item())
        result["ntk_max_eigenvalue"] = float(kernel_eigenvalues.max().item())
    if residual is not None:
        error = residual.detach().float().reshape(-1)
        if error.numel() == rows:
            vector = jacobian.detach().float() @ error
            result["residual_alignment"] = float(torch.linalg.vector_norm(vector).item())
            result["residual_norm"] = float(torch.linalg.vector_norm(error).item())
    return result


def sampled_parameter_jacobian(
    model: Any,
    inputs: Any,
    *,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    max_samples: int = 16,
) -> Any:
    """Build a scalar-output Jacobian for a bounded sample subset.

    The scalar per-sample output is the mean of the model output.  This is a
    transparent low-cost probe, not a full logits Jacobian or exact Hessian.
    ``forward_fn`` is useful for EEG models whose batch is a tuple.
    """

    torch = _torch()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    if forward_fn is None:
        forward_fn = lambda module, batch: module(batch)
    list_input = False
    if isinstance(inputs, torch.Tensor):
        samples = [inputs[index : index + 1] for index in range(min(int(inputs.shape[0]), max_samples))]
    elif isinstance(inputs, (tuple, list)) and inputs and all(isinstance(item, torch.Tensor) for item in inputs):
        count = min(int(inputs[0].shape[0]), max_samples)
        list_input = isinstance(inputs, list)
        samples = [
            ([item[index : index + 1] for item in inputs] if list_input else tuple(item[index : index + 1] for item in inputs))
            for index in range(count)
        ]
    else:
        raise TypeError("inputs must be a tensor or a tuple/list of tensors")
    rows: list[Any] = []
    was_training = bool(model.training)
    model.eval()
    try:
        for sample in samples:
            output = forward_fn(model, sample)
            if not isinstance(output, torch.Tensor):
                raise TypeError("forward_fn must return a tensor")
            scalar = output.float().reshape(-1).mean()
            gradients = torch.autograd.grad(scalar, parameters, retain_graph=False, allow_unused=True)
            rows.append(torch.cat([
                (gradient if gradient is not None else torch.zeros_like(parameter)).detach().float().reshape(-1)
                for parameter, gradient in zip(parameters, gradients)
            ]).cpu())
    finally:
        model.train(was_training)
    if not rows:
        return torch.empty((0, sum(int(parameter.numel()) for parameter in parameters)))
    return torch.stack(rows)


def local_linearity_summary(
    model: Any,
    inputs: Any,
    *,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    epsilons: Sequence[float] = (1e-4, 1e-3, 1e-2),
    directions: int = 4,
    seed: int = 20260827,
) -> dict[str, Any]:
    """Measure multi-direction finite-difference Taylor remainders."""

    torch = _torch()
    if forward_fn is None:
        forward_fn = lambda module, batch: module(batch)
    if directions < 1:
        raise ValueError("directions must be positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    probe = copy.deepcopy(model).eval()
    probe_inputs = inputs
    generator = torch.Generator(device=parameters[0].device.type).manual_seed(int(seed))
    base = forward_fn(probe, probe_inputs).detach().float()
    parameter_norm = math.sqrt(sum(float(parameter.detach().float().square().sum().item()) for parameter in probe.parameters()))
    rows: list[dict[str, float | int]] = []
    for direction_index in range(directions):
        direction = [torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype, device=parameter.device) for parameter in parameters]
        direction_norm = math.sqrt(sum(float(item.detach().float().square().sum().item()) for item in direction))
        scale = max(parameter_norm, 1.0) / max(direction_norm, EPS)
        direction = [item * scale for item in direction]
        for epsilon in epsilons:
            if epsilon <= 0:
                raise ValueError("epsilons must be positive")
            with torch.no_grad():
                for parameter, delta in zip(parameters, direction):
                    parameter.add_(delta * float(epsilon))
                first = forward_fn(probe, probe_inputs).detach().float()
                for parameter, delta in zip(parameters, direction):
                    parameter.add_(delta * float(epsilon))
                second = forward_fn(probe, probe_inputs).detach().float()
                for parameter, delta in zip(parameters, direction):
                    parameter.sub_(delta * float(2.0 * epsilon))
            first_difference = first - base
            second_difference = second - 2.0 * first + base
            first_norm = torch.linalg.vector_norm(first_difference)
            second_norm = torch.linalg.vector_norm(second_difference)
            rows.append({
                "direction": direction_index,
                "epsilon": float(epsilon),
                "first_difference_norm": float(first_norm.item()),
                "second_difference_norm": float(second_norm.item()),
                "relative_second_difference": float((second_norm / first_norm.clamp_min(EPS)).item()),
            })
    ratios = [float(row["relative_second_difference"]) for row in rows]
    return {
        "status": "computed",
        "seed": int(seed),
        "direction_count": int(directions),
        "epsilons": [float(value) for value in epsilons],
        "rows": rows,
        "mean_relative_second_difference": sum(ratios) / len(ratios),
        "max_relative_second_difference": max(ratios),
        "interpretation": "finite-difference local diagnostic; not a global linearity proof",
    }


def parameter_norm_summary(model: Any, reference: Any | None = None) -> dict[str, Any]:
    """Return block-independent parameter norms and optional state deltas."""

    torch = _torch()
    current = {name: parameter.detach().float() for name, parameter in model.named_parameters()}
    reference_values = {name: parameter.detach().float() for name, parameter in reference.named_parameters()} if reference is not None else {}
    rows: list[dict[str, Any]] = []
    total_sq = 0.0
    delta_sq = 0.0
    reference_sq = 0.0
    for name, value in current.items():
        norm = float(torch.linalg.vector_norm(value).item())
        total_sq += norm * norm
        row: dict[str, Any] = {"name": name, "l2": norm, "parameter_count": int(value.numel())}
        if name in reference_values and reference_values[name].shape == value.shape:
            delta = value - reference_values[name]
            delta_norm = float(torch.linalg.vector_norm(delta).item())
            ref_norm = float(torch.linalg.vector_norm(reference_values[name]).item())
            delta_sq += delta_norm * delta_norm
            reference_sq += ref_norm * ref_norm
            row.update({"delta_l2": delta_norm, "reference_l2": ref_norm, "relative_update": delta_norm / max(ref_norm, EPS)})
        rows.append(row)
    result: dict[str, Any] = {
        "status": "computed",
        "data_dependent": False,
        "global_l2": math.sqrt(total_sq),
        "parameter_count": sum(int(value.numel()) for value in current.values()),
        "parameters": rows,
    }
    if reference is not None:
        result.update({
            "global_delta_l2": math.sqrt(delta_sq),
            "global_reference_l2": math.sqrt(reference_sq),
            "global_relative_update": math.sqrt(delta_sq) / max(math.sqrt(reference_sq), EPS),
            "reference_provided": True,
        })
    else:
        result["reference_provided"] = False
    return result


def parameter_spectral_summary(model: Any, *, max_singular_values: int = 16) -> dict[str, Any]:
    """Summarize singular-value geometry of trainable weight tensors.

    Convolutional kernels are reshaped to ``[out_channels, -1]``; linear
    weights already have that orientation.  Bias and one-dimensional scale
    parameters are reported as ``skipped`` because assigning them a matrix
    spectrum would be arbitrary.  No input batch is consumed, so this is a
    checkpoint/state diagnostic rather than a data-dependent representation
    metric.
    """

    torch = _torch()
    rows: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        value = parameter.detach().float()
        if value.ndim < 2:
            rows[name] = {
                "status": "skipped",
                "reason": "parameter has fewer than two dimensions",
                "shape": [int(item) for item in value.shape],
                "data_dependent": False,
            }
            continue
        matrix = value.reshape(int(value.shape[0]), -1)
        summary = spectral_summary(matrix, feature_axis=1, center=False)
        singular = torch.linalg.svdvals(matrix)
        limit = max(0, int(max_singular_values))
        summary.update(
            {
                "status": "computed",
                "tensor_shape": [int(item) for item in value.shape],
                "matrix_shape": [int(item) for item in matrix.shape],
                "top_singular_values": singular[:limit].tolist() if limit else [],
                "data_dependent": False,
            }
        )
        rows[name] = summary
    return {
        "status": "computed",
        "data_dependent": False,
        "max_singular_values": int(max_singular_values),
        "parameters": rows,
    }


def gradient_summary(gradients: Any) -> dict[str, Any]:
    """Summarize a ``[batches, parameters]`` gradient matrix."""

    torch = _torch()
    if not isinstance(gradients, torch.Tensor) or gradients.ndim != 2:
        raise ValueError("gradients must be a rank-2 tensor")
    if gradients.shape[0] == 0:
        return {"status": "empty"}
    values = gradients.detach().float()
    norms = torch.linalg.vector_norm(values, dim=1)
    if values.shape[0] > 1:
        normalized = values / norms[:, None].clamp_min(EPS)
        cosine = normalized @ normalized.T
        offdiag = cosine[~torch.eye(values.shape[0], dtype=torch.bool, device=values.device)]
        cosine_mean = float(offdiag.mean().item()) if offdiag.numel() else None
        negative = float((offdiag < 0).float().mean().item()) if offdiag.numel() else None
    else:
        cosine_mean = None
        negative = None
    return {
        "status": "computed",
        "data_dependent": True,
        "batch_count": int(values.shape[0]),
        "parameter_dim": int(values.shape[1]),
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std(unbiased=False).item()),
        "cosine_mean": cosine_mean,
        "negative_fraction": negative,
        "effective_rank": spectral_summary(values, feature_axis=1, center=True)["effective_rank"],
    }


def capture_representations(
    model: Any,
    batches: Iterable[Any],
    layer_map: Mapping[str, Any],
    *,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    max_batches: int = 0,
) -> dict[str, Any]:
    """Capture named module outputs using forward hooks without model edits."""

    torch = _torch()
    if forward_fn is None:
        forward_fn = lambda module, batch: module(batch)
    collected: dict[str, list[Any]] = {name: [] for name in layer_map}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                collected[name].append(output.detach().cpu())
            elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                collected[name].append(output[0].detach().cpu())
        return hook

    for name, module in layer_map.items():
        handles.append(module.register_forward_hook(make_hook(name)))
    was_training = bool(model.training)
    model.eval()
    count = 0
    try:
        with torch.no_grad():
            for batch in batches:
                if max_batches > 0 and count >= max_batches:
                    break
                forward_fn(model, batch)
                count += 1
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    result: dict[str, Any] = {}
    for name, values in collected.items():
        if not values:
            result[name] = None
        else:
            result[name] = torch.cat(values, dim=0)
    return result


def _move_batch_value(value: Any, device: Any) -> Any:
    """Recursively move tensor/multi-input payloads for probe helpers."""

    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_batch_value(item, device) for item in value)
    if isinstance(value, list):
        return [_move_batch_value(item, device) for item in value]
    if isinstance(value, Mapping):
        return {key: _move_batch_value(item, device) for key, item in value.items()}
    return value


def _extract_logits_value(output: Any) -> Any:
    """Extract a tensor from tensor/bundle/mapping model outputs."""

    torch = _torch()
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return _extract_logits_value(output.logits)
    if isinstance(output, Mapping):
        for key in ("logits", "output", "outputs", "prediction", "pred"):
            if key in output:
                try:
                    return _extract_logits_value(output[key])
                except TypeError:
                    continue
    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return _extract_logits_value(item)
            except TypeError:
                continue
    raise TypeError("model output does not contain a logits tensor")


def evaluate_classifier(model: Any, batches: Iterable[Any], *, forward_fn: Callable[[Any, Any], Any] | None = None) -> dict[str, float]:
    """Evaluate a classifier on ``(inputs, labels)`` batches."""

    torch = _torch()
    import torch.nn.functional as F

    if forward_fn is None:
        forward_fn = lambda module, inputs: module(inputs)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    was_training = bool(model.training)
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in batches:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise ValueError("evaluation batches must be (inputs, labels)")
            inputs, labels = batch
            inputs = _move_batch_value(inputs, model_device)
            labels = labels.to(model_device)
            logits = _extract_logits_value(forward_fn(model, inputs))
            if logits.ndim > 2:
                logits = logits.reshape(-1, logits.shape[-1])
                labels = labels.reshape(-1)
            loss = F.cross_entropy(logits, labels.long())
            loss_sum += float(loss.item()) * int(labels.numel())
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
    model.train(was_training)
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1), "count": float(total)}


def _clone_model(model: Any) -> Any:
    return copy.deepcopy(model)


def fixed_budget_probe(
    warm_model: Any,
    fresh_model: Any,
    train_batches: Sequence[Any],
    eval_batches: Sequence[Any],
    *,
    steps: Sequence[int] = (0, 5, 10, 25),
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    seed: int = 20260827,
    freeze_batch_norm: bool = True,
) -> dict[str, Any]:
    """Compare warm and fresh models under identical finite update budget."""

    torch = _torch()
    if not train_batches or not eval_batches:
        raise ValueError("train_batches and eval_batches must be non-empty")
    requested = sorted({int(step) for step in steps})
    if not requested or requested[0] != 0 or requested[-1] < 1:
        raise ValueError("steps must include 0 and at least one positive step")
    torch.manual_seed(int(seed))
    warm = _clone_model(warm_model)
    fresh = _clone_model(fresh_model)
    if forward_fn is None:
        forward_fn = lambda module, inputs: module(inputs)
    warm_optimizer = torch.optim.Adam(warm.parameters(), lr=lr, weight_decay=weight_decay)
    fresh_optimizer = torch.optim.Adam(fresh.parameters(), lr=lr, weight_decay=weight_decay)
    curves: dict[str, list[dict[str, float]]] = {"warm": [], "fresh": []}

    def record(name: str, model: Any, step: int) -> None:
        metrics = evaluate_classifier(model, eval_batches, forward_fn=forward_fn)
        curves[name].append({"step": float(step), "loss": metrics["loss"], "accuracy": metrics["accuracy"]})

    record("warm", warm, 0)
    record("fresh", fresh, 0)
    train_index = 0
    for step in range(1, requested[-1] + 1):
        batch = train_batches[train_index % len(train_batches)]
        train_index += 1
        for model, optimizer in ((warm, warm_optimizer), (fresh, fresh_optimizer)):
            model.train()
            if freeze_batch_norm:
                # BatchNorm running-stat updates are stateful and can make a
                # warm-vs-fresh comparison depend on batch order rather than
                # on parameter plasticity.  Keep the policy explicit.
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
            inputs, labels = batch
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = torch.device("cpu")
            inputs = _move_batch_value(inputs, model_device)
            labels = labels.to(model_device)
            logits = _extract_logits_value(forward_fn(model, inputs))
            if logits.ndim > 2:
                logits = logits.reshape(-1, logits.shape[-1])
                labels = labels.reshape(-1)
            import torch.nn.functional as F
            loss = F.cross_entropy(logits, labels.long())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if step in requested:
            record("warm", warm, step)
            record("fresh", fresh, step)
    warm_curve = curves["warm"]
    fresh_curve = curves["fresh"]
    warm_acc = warm_curve[-1]["accuracy"]
    fresh_acc = fresh_curve[-1]["accuracy"]
    warm_loss = warm_curve[-1]["loss"]
    fresh_loss = fresh_curve[-1]["loss"]

    def auc(curve: list[dict[str, float]], field: str) -> float:
        if len(curve) == 1 or curve[-1]["step"] <= 0:
            return curve[0][field]
        area = 0.0
        for left, right in zip(curve, curve[1:]):
            area += (right["step"] - left["step"]) * (left[field] + right[field]) / 2.0
        return area / curve[-1]["step"]

    return {
        "protocol": "fixed-budget-fresh-vs-warm-heldout-v1",
        "seed": int(seed),
        "steps": requested,
        "freeze_batch_norm": bool(freeze_batch_norm),
        "curves": curves,
        "outcome": {
            "fresh_gap_final": fresh_acc - warm_acc,
            "fresh_loss_gap_final": warm_loss - fresh_loss,
            "fresh_auc_gap": auc(fresh_curve, "accuracy") - auc(warm_curve, "accuracy"),
            "fresh_loss_auc_gap": auc(warm_curve, "loss") - auc(fresh_curve, "loss"),
            "warm_acc_gain": warm_acc - warm_curve[0]["accuracy"],
            "fresh_acc_gain": fresh_acc - fresh_curve[0]["accuracy"],
        },
    }


# Curvature estimators and the unified model diagnostic live in a dedicated
# module, but are re-exported here for backwards compatibility with callers
# that already import ``edgeforge.lop_metrics``.  ``lop_diagnostics`` imports
# this module lazily inside ``diagnose_model`` so the two modules remain safe
# to import in either order (and PyTorch stays optional at import time).
from .lop_diagnostics import (  # noqa: E402  (intentional compatibility export)
    calibration_manifest_digest,
    calibration_manifest_provenance,
    DiagnosticsConfig,
    LoPDiagnostics,
    LoPMetricConfig,
    MetricConfig,
    diagnose_model,
    empirical_fisher_diagonal,
    empirical_fisher_summary,
    estimate_hessian_trace,
    exact_hessian_matrix,
    fisher_diagonal,
    fisher_summary,
    hessian_hvp,
    hessian_top_eigenvalue,
    hessian_top_eigenvalue_summary,
    hessian_vector_product,
    hessian_trace,
    hutchinson_trace,
    hutchinson_trace_summary,
    hvp,
    power_iteration_top_eigenvalue,
    split_batch,
    top_hessian_eigenvalue,
)
