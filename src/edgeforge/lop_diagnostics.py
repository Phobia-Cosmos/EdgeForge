"""Model-independent LoP diagnostics and curvature estimators.

This module contains the slightly more expensive, data-dependent probes that
are intentionally kept separate from :mod:`edgeforge.lop_metrics`' small
tensor summaries.  PyTorch remains an optional dependency: importing this
module does not import torch, and all numerical helpers load it lazily.

The central entry point is :func:`diagnose_model`.  It accepts a PyTorch model
and a finite calibration batch sequence and returns a JSON-serialisable
dictionary containing representation, Jacobian/NTK, curvature and gradient
diagnostics.  The helpers are descriptive probes; they do not establish a
causal ``loss of plasticity`` claim by themselves.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EPS = 1e-12


def _sha256_bytes(value: bytes) -> str:
    """Return a stable, unprefixed SHA-256 digest for provenance records."""

    return hashlib.sha256(value).hexdigest()


def calibration_manifest_digest(manifest: Any) -> str | None:
    """Hash a calibration manifest without touching model/data payloads.

    ``manifest`` may be a filesystem path, raw bytes, or an in-memory
    mapping.  Paths are hashed byte-for-byte so the digest binds the exact
    manifest used by an experiment; mappings use canonical JSON encoding.
    Missing paths return ``None`` rather than aborting a diagnostic run.  The
    helper intentionally does not recurse into paths listed *inside* a JSON
    manifest, so no EEG samples are read or uploaded as a side effect.
    """

    if manifest is None:
        return None
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        try:
            return _sha256_bytes(path.read_bytes()) if path.is_file() else None
        except OSError:
            return None
    if isinstance(manifest, bytes):
        return _sha256_bytes(manifest)
    if isinstance(manifest, Mapping):
        try:
            encoded = json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError):
            return None
        return _sha256_bytes(encoded)
    return None


def calibration_manifest_provenance(
    manifest: Any,
    *,
    declared_digest: str | None = None,
) -> dict[str, Any]:
    """Describe calibration-manifest identity for a JSON report.

    The returned object is deliberately explicit about missing/unreadable
    manifests.  A declared digest is retained even when the local file is
    unavailable, allowing a remote Worker to verify it later; when both are
    available ``matches_declared`` exposes a cheap integrity check.
    """

    declared = str(declared_digest) if declared_digest is not None else None
    result: dict[str, Any] = {
        "status": "not-provided" if manifest is None and declared is None else "missing",
        "path": str(manifest) if isinstance(manifest, (str, Path)) else None,
        "sha256": None,
        "declared_sha256": declared,
        "bytes": None,
        "matches_declared": None,
    }
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        try:
            if path.is_file():
                raw = path.read_bytes()
                digest = _sha256_bytes(raw)
                result.update(
                    {
                        "status": "present",
                        "sha256": digest,
                        "bytes": len(raw),
                    }
                )
                if declared is not None:
                    result["matches_declared"] = digest == declared.removeprefix("sha256:")
        except OSError as error:
            result["error"] = f"{type(error).__name__}: {error}"
        return result
    digest = calibration_manifest_digest(manifest)
    if digest is not None:
        encoded_size: int | None = None
        if isinstance(manifest, bytes):
            encoded_size = len(manifest)
        elif isinstance(manifest, Mapping):
            try:
                encoded_size = len(
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                )
            except (TypeError, ValueError):
                encoded_size = None
        result.update({"status": "inline", "sha256": digest, "bytes": encoded_size})
        if declared is not None:
            result["matches_declared"] = digest == declared.removeprefix("sha256:")
    return result


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised without torch
        raise RuntimeError("PyTorch is required for LoP numerical diagnostics") from error
    return torch


@dataclass
class MetricConfig:
    """Configuration shared by :func:`diagnose_model` and curvature probes.

    The defaults are deliberately conservative so a diagnostic run can be
    used in CI or on an edge device.  ``max_batches`` bounds data traversal;
    ``max_observations`` is forwarded to SVD/CKA summaries.  Set a count to
    zero to disable that particular bound.

    ``objective`` controls the loss used for gradients and Hessian probes.
    The built-in objectives are ``cross_entropy`` (requires labels),
    ``mse``/``mse_to_zero`` (labels optional), and ``output_mean``.  A custom
    callable can be passed to :func:`diagnose_model` via ``loss_fn``.
    """

    max_observations: int = 256
    jacobian_samples: int = 4
    hessian_mode: str = "hvp"
    hessian_probes: int = 8
    hessian_iterations: int = 20
    hessian_tolerance: float = 1e-5
    objective: str = "cross_entropy"
    label_source: str = "true"
    freeze_batch_norm: bool = True
    max_batches: int = 8
    seed: int = 20260827
    include_representations: bool = True
    include_jacobian: bool = True
    include_hessian: bool = True
    include_gradients: bool = True
    include_fisher: bool = False
    fisher_samples: int = 32
    exact_hessian_max_params: int = 256
    feature_axes: dict[str, int] = field(default_factory=dict)
    calibration_manifest: Any = None
    calibration_manifest_digest: str | None = None
    fresh_warm: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        integer_fields = (
            "max_observations",
            "jacobian_samples",
            "hessian_probes",
            "hessian_iterations",
            "max_batches",
            "fisher_samples",
            "exact_hessian_max_params",
        )
        for name in integer_fields:
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            setattr(self, name, value)
        self.hessian_tolerance = float(self.hessian_tolerance)
        if self.hessian_tolerance <= 0:
            raise ValueError("hessian_tolerance must be positive")
        self.hessian_mode = str(self.hessian_mode).lower().strip()
        valid_modes = {"none", "hvp", "power", "hutchinson", "exact", "all"}
        if self.hessian_mode not in valid_modes:
            raise ValueError(f"hessian_mode must be one of {sorted(valid_modes)}")
        self.objective = str(self.objective).lower().strip()
        self.label_source = str(self.label_source).lower().strip()
        if self.label_source not in {"true", "pseudo", "none"}:
            raise ValueError("label_source must be 'true', 'pseudo', or 'none'")
        self.feature_axes = {str(key): int(value) for key, value in dict(self.feature_axes).items()}
        # Keep inline mappings/bytes intact so a caller can bind a generated
        # manifest without first writing a temporary file.  Filesystem-like
        # values are normalized to strings for stable JSON output.
        if isinstance(self.calibration_manifest, (str, Path)):
            self.calibration_manifest = str(self.calibration_manifest)
        if self.calibration_manifest_digest is not None:
            self.calibration_manifest_digest = str(self.calibration_manifest_digest)
        if self.fresh_warm is None:
            self.fresh_warm = {}
        elif not isinstance(self.fresh_warm, Mapping):
            raise TypeError("fresh_warm must be a mapping")
        else:
            self.fresh_warm = dict(self.fresh_warm)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for JSON serialisation."""

        result = dataclasses.asdict(self)
        manifest = self.calibration_manifest
        if isinstance(manifest, bytes):
            # Do not place raw manifest bytes in experiment metadata; retain a
            # verifiable identity and size instead.
            result["calibration_manifest"] = {
                "inline_sha256": _sha256_bytes(manifest),
                "bytes": len(manifest),
            }
        elif isinstance(manifest, Path):
            result["calibration_manifest"] = str(manifest)
        return result

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None, **overrides: Any) -> "MetricConfig":
        """Build a config from JSON/YAML-like mappings.

        A few explicit aliases make experiment manifests less brittle while
        preserving strict validation for genuinely unknown options.
        """

        merged = dict(values or {})
        merged.update(overrides)
        # Accept the nested form used by experiment YAML manifests while
        # storing one flat, serialisable dataclass internally.
        hessian = merged.pop("hessian", None)
        if isinstance(hessian, Mapping):
            nested_aliases = {
                "mode": "hessian_mode",
                "probes": "hessian_probes",
                "num_probes": "hessian_probes",
                "iterations": "hessian_iterations",
                "max_iter": "hessian_iterations",
                "tolerance": "hessian_tolerance",
                "tol": "hessian_tolerance",
            }
            for source, target in nested_aliases.items():
                if source in hessian and target not in merged:
                    merged[target] = hessian[source]
        elif hessian is not None:
            raise TypeError("hessian must be a mapping when supplied")
        batchnorm = merged.pop("batchnorm", None)
        if batchnorm is None:
            batchnorm = merged.pop("batch_norm", None)
        if isinstance(batchnorm, Mapping):
            for source, target in {
                "freeze_running_stats": "freeze_batch_norm",
                "freeze": "freeze_batch_norm",
            }.items():
                if source in batchnorm and target not in merged:
                    merged[target] = batchnorm[source]
        elif batchnorm is not None:
            raise TypeError("batchnorm must be a mapping when supplied")
        objective = merged.get("objective")
        if isinstance(objective, Mapping):
            objective_mapping = dict(objective)
            merged["objective"] = objective_mapping.get("name", objective_mapping.get("loss", "cross_entropy"))
            if "label_source" in objective_mapping and "label_source" not in merged:
                merged["label_source"] = objective_mapping["label_source"]
        if "label_source" in merged:
            label_aliases = {
                "ground_truth": "true",
                "ground-truth": "true",
                "pseudo_label": "pseudo",
                "pseudo-label": "pseudo",
                "unlabeled": "none",
                "none": "none",
            }
            merged["label_source"] = label_aliases.get(str(merged["label_source"]).lower(), merged["label_source"])
        aliases = {
            "max_jacobian_samples": "jacobian_samples",
            "jacobian_max_samples": "jacobian_samples",
            "hessian_num_probes": "hessian_probes",
            "num_hessian_probes": "hessian_probes",
            "power_iterations": "hessian_iterations",
            "hessian_max_iter": "hessian_iterations",
            "hessian_tol": "hessian_tolerance",
            "loss": "objective",
            "fisher_max_samples": "fisher_samples",
        }
        for source, target in aliases.items():
            if source in merged and target not in merged:
                merged[target] = merged.pop(source)
        valid = {item.name for item in dataclasses.fields(cls)}
        unknown = sorted(set(merged) - valid)
        if unknown:
            raise TypeError(f"unknown MetricConfig field(s): {', '.join(unknown)}")
        return cls(**merged)


# Descriptive aliases used by experiment manifests and older notebooks.
LoPMetricConfig = MetricConfig
DiagnosticsConfig = MetricConfig


def _is_tensor(value: Any) -> bool:
    torch = _torch()
    return isinstance(value, torch.Tensor)


def _looks_like_labels(value: Any, input_value: Any) -> bool:
    """Heuristically distinguish ``(inputs, labels)`` from multi-input data.

    EEG models frequently receive ``(eeg, eog)`` tensors.  Treating the second
    tensor as labels would silently build a nonsensical Hessian objective, so
    only low-rank tensors (or integer tensors) are interpreted as targets.
    """

    torch = _torch()
    if not isinstance(value, torch.Tensor):
        # Scalar/list targets from lightweight datasets are valid labels.  A
        # nested tuple/list of tensors is handled as a multi-input payload by
        # the caller and should not be classified here.
        return isinstance(value, (int, float, bool)) or (
            isinstance(value, (list, tuple))
            and not any(isinstance(item, torch.Tensor) and item.ndim >= 3 for item in value)
        )
    if value.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
        torch.bool,
    }:
        return True
    if not isinstance(input_value, torch.Tensor):
        return value.ndim <= 2
    # Classification/regression targets are conventionally [B], [B, 1], or
    # [B, K] (one-hot).  EEG multi-input payloads, in contrast, generally have
    # at least a channel/time/spatial axis and are rank >= 3.  Rank <= 2 is
    # therefore the safest practical boundary, including MSE targets with the
    # same shape as a [B, 1] input.
    return value.ndim <= 2


def split_batch(batch: Any) -> tuple[Any, Any | None]:
    """Split a batch into model inputs and optional labels.

    Mapping batches accept common aliases (``inputs``/``x``/``data`` and
    ``labels``/``y``/``target``).  A two-item tuple is considered labelled only
    when its second item looks like a target; otherwise it remains a
    multi-input tuple.
    """

    if isinstance(batch, Mapping):
        input_value = next((batch[key] for key in ("inputs", "input", "x", "data") if key in batch), None)
        if input_value is None:
            # Preserve custom mapping batches for a user-supplied forward_fn.
            input_value = batch
        labels = next((batch[key] for key in ("labels", "label", "y", "target", "targets") if key in batch), None)
        return input_value, labels
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        first, second = batch
        # A list of two tensors is the conventional representation emitted by
        # several lightweight collators for a *multi-input* batch (e.g.
        # ``[eeg, eog]``).  Both values can be rank-2 in small smoke tests,
        # which makes a shape-only label heuristic inherently ambiguous.  In
        # that case preserve the list as model inputs; callers that need a
        # labelled pair should use the unambiguous tuple ``(x, y)`` or a
        # mapping with a ``labels``/``y`` key.
        if isinstance(batch, list) and all(_is_tensor(item) for item in batch):
            return batch, None
        if _looks_like_labels(second, first):
            return first, second
    return batch, None


def _materialize_batches(batches: Iterable[Any] | Any, max_batches: int = 0) -> list[Any]:
    """Materialise a bounded batch sequence without splitting a single pair."""

    torch = _torch()
    if isinstance(batches, torch.Tensor) or isinstance(batches, Mapping):
        values = [batches]
    elif isinstance(batches, (tuple, list)) and len(batches) == 2 and all(isinstance(item, torch.Tensor) for item in batches):
        # A bare pair of tensors is commonly a single multi-input batch (for
        # example ``(eeg, eog)``), but it is also how a labelled batch is
        # represented.  Keep the pair intact so ``split_batch`` can apply its
        # target-shape heuristic in either case.  The list form matters for
        # callers that materialise a dataloader batch as ``[eeg, eog]``.
        values = [batches]
    elif isinstance(batches, (tuple, list)) and len(batches) == 2 and _looks_like_labels(batches[1], batches[0]):
        values = [batches]
    else:
        try:
            values = list(batches)
        except TypeError:
            values = [batches]
    if max_batches > 0:
        values = values[:max_batches]
    if not values:
        raise ValueError("batches must contain at least one item")
    return values


def _model_device(model: Any):
    torch = _torch()
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _move_to_device(value: Any, device: Any) -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _default_forward(model: Any, inputs: Any) -> Any:
    """Call a model with tensor or multi-input payloads."""

    if isinstance(inputs, tuple):
        try:
            return model(inputs)
        except (TypeError, ValueError):
            return model(*inputs)
    if isinstance(inputs, list):
        try:
            return model(inputs)
        except (TypeError, ValueError):
            return model(*inputs)
    if isinstance(inputs, Mapping):
        try:
            return model(**inputs)
        except (TypeError, ValueError):
            return model(inputs)
    return model(inputs)


def _invoke_forward(model: Any, inputs: Any, forward_fn: Callable[[Any, Any], Any] | None) -> Any:
    return (forward_fn(model, inputs) if forward_fn is not None else _default_forward(model, inputs))


def _extract_tensor(output: Any) -> Any:
    """Extract logits/tensor from common model output containers."""

    torch = _torch()
    if isinstance(output, torch.Tensor):
        return output
    # ``EEGForwardBundle`` (and similar dataclass/named objects) exposes
    # logits as an attribute rather than a mapping key.  Importing the class
    # here would couple the generic diagnostics module to EEG models, so use a
    # small structural check instead.
    if hasattr(output, "logits"):
        try:
            return _extract_tensor(getattr(output, "logits"))
        except TypeError:
            pass
    if isinstance(output, Mapping):
        for key in ("logits", "output", "outputs", "pred", "prediction"):
            if key in output:
                try:
                    return _extract_tensor(output[key])
                except TypeError:
                    pass
        for value in output.values():
            try:
                return _extract_tensor(value)
            except TypeError:
                continue
    if isinstance(output, (tuple, list)):
        for value in output:
            try:
                return _extract_tensor(value)
            except TypeError:
                continue
    raise TypeError("model output does not contain a torch.Tensor")


def _bundle_representations(output: Any) -> Mapping[str, Any] | None:
    """Return named representations from a forward bundle, if present."""

    values = getattr(output, "representations", None)
    if isinstance(values, Mapping):
        return values
    if isinstance(output, Mapping) and isinstance(output.get("representations"), Mapping):
        return output["representations"]
    return None


def _invoke_loss_fn(loss_fn: Callable[..., Any], model: Any, batch: Any, logits: Any, labels: Any | None) -> Any:
    """Invoke custom loss functions with a small, backwards-compatible API."""

    # Prefer the conventional ``loss_fn(logits, labels)`` signature.  A
    # caller that needs the complete batch/model can use either of the two
    # fallbacks.  Signature inspection avoids swallowing TypeError raised
    # inside a valid loss implementation.
    try:
        signature = inspect.signature(loss_fn)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(parameter.kind == parameter.VAR_POSITIONAL for parameter in signature.parameters.values())
        count = len(positional)
    except (TypeError, ValueError):
        count, has_varargs = 2, False
    if has_varargs or count <= 2:
        return loss_fn(logits, labels)
    if count == 3:
        return loss_fn(model, batch, logits)
    return loss_fn(model, batch, logits, labels)


def _objective_loss(
    model: Any,
    batch: Any,
    logits: Any,
    labels: Any | None,
    *,
    objective: str,
    label_source: str = "true",
    loss_fn: Callable[..., Any] | None = None,
) -> Any:
    torch = _torch()
    import torch.nn.functional as F

    if loss_fn is not None:
        value = _invoke_loss_fn(loss_fn, model, batch, logits, labels)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=logits.dtype, device=logits.device)
        return value.float().mean()
    objective = str(objective).lower().strip()
    if label_source == "pseudo" and labels is None:
        labels = logits.detach().argmax(dim=-1)
    if objective in {"cross_entropy", "ce", "nll", "negative_log_likelihood"}:
        if labels is None:
            raise ValueError("cross_entropy objective requires labels (or label_source='pseudo')")
        values = logits
        targets = labels
        if values.ndim > 2:
            values = values.reshape(-1, values.shape[-1])
            targets = targets.reshape(-1)
        if targets.ndim > 1 and targets.shape[-1] == values.shape[-1]:
            targets = targets.argmax(dim=-1)
        return F.cross_entropy(values, targets.long())
    if objective in {"mse", "mean_squared_error"}:
        if labels is None:
            raise ValueError("mse objective requires labels")
        target = labels.to(device=logits.device, dtype=logits.dtype)
        return F.mse_loss(logits, target)
    if objective in {"mse_to_zero", "output_mean_square", "squared_output"}:
        return logits.float().square().mean()
    if objective in {"output_mean", "mean_output"}:
        return logits.float().mean()
    if objective in {"output_norm", "norm"}:
        return logits.float().square().sum(dim=-1).sqrt().mean()
    raise ValueError(f"unsupported objective: {objective}")


def objective_loss(
    model: Any,
    batch: Any,
    logits: Any,
    labels: Any | None,
    *,
    objective: str = "cross_entropy",
    label_source: str = "true",
    loss_fn: Callable[..., Any] | None = None,
) -> Any:
    """Compute the configured diagnostic objective through the shared path.

    This public wrapper keeps experiment adapters from duplicating the
    objective semantics used by Hessian, Fisher and gradient probes.  The
    returned scalar remains data/objective dependent; callers should pass
    ``labels=None`` with ``label_source='pseudo'`` when pseudo labels must be
    derived from the current logits rather than supplied annotations.
    """

    return _objective_loss(
        model,
        batch,
        logits,
        labels,
        objective=objective,
        label_source=label_source,
        loss_fn=loss_fn,
    )


def _average_loss(
    model: Any,
    batches: Sequence[Any],
    *,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    objective: str = "cross_entropy",
    label_source: str = "true",
    loss_fn: Callable[..., Any] | None = None,
) -> Any:
    torch = _torch()
    device = _model_device(model)
    losses: list[Any] = []
    for batch in batches:
        inputs, labels = split_batch(batch)
        inputs = _move_to_device(inputs, device)
        if labels is not None:
            labels = _move_to_device(labels, device)
        output = _invoke_forward(model, inputs, forward_fn)
        logits = _extract_tensor(output)
        losses.append(_objective_loss(model, batch, logits, labels, objective=objective, label_source=label_source, loss_fn=loss_fn))
    if not losses:
        raise ValueError("batches must contain at least one item")
    return torch.stack([loss if loss.ndim == 0 else loss.mean() for loss in losses]).mean()


def _trainable_parameters(model: Any) -> list[Any]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _flatten(values: Sequence[Any], *, device: Any | None = None, dtype: Any | None = None) -> Any:
    torch = _torch()
    chunks = []
    for value in values:
        item = value.reshape(-1)
        if device is not None or dtype is not None:
            item = item.to(device=device if device is not None else item.device, dtype=dtype or item.dtype)
        chunks.append(item)
    if not chunks:
        return torch.empty(0, device=device, dtype=dtype or torch.float32)
    return torch.cat(chunks)


def _split_vector(vector: Any, parameters: Sequence[Any]) -> list[Any]:
    torch = _torch()
    total = sum(int(parameter.numel()) for parameter in parameters)
    if vector.numel() != total:
        raise ValueError(f"vector has {vector.numel()} elements, expected {total}")
    result: list[Any] = []
    offset = 0
    for parameter in parameters:
        count = int(parameter.numel())
        result.append(vector[offset : offset + count].reshape_as(parameter).to(device=parameter.device, dtype=parameter.dtype))
        offset += count
    return result


def _coerce_flat_vector(vector: Any, parameters: Sequence[Any]) -> Any:
    """Convert a flat or per-parameter direction to one contiguous tensor."""

    torch = _torch()
    if not parameters:
        return torch.empty(0)
    if isinstance(vector, (tuple, list)) and len(vector) == len(parameters):
        return _flatten(
            [
                torch.as_tensor(item, device=parameter.device, dtype=parameter.dtype).reshape_as(parameter)
                for parameter, item in zip(parameters, vector)
            ],
            device=parameters[0].device,
            dtype=parameters[0].dtype,
        )
    return torch.as_tensor(vector, device=parameters[0].device, dtype=parameters[0].dtype).reshape(-1)


def hessian_vector_product(
    model: Any,
    batches: Iterable[Any] | Any,
    vector: Any,
    *,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    objective: str = "cross_entropy",
    label_source: str = "true",
    loss_fn: Callable[..., Any] | None = None,
) -> Any:
    """Compute an exact Hessian-vector product for a calibration objective.

    The Hessian is with respect to trainable model parameters and the loss is
    averaged over ``batches``.  Inputs (and labels for supervised objectives)
    therefore affect the result; architecture and current weights alone are
    insufficient to determine it.  No parameter ``.grad`` fields are mutated.
    """

    torch = _torch()
    parameters = _trainable_parameters(model)
    if not parameters:
        raise ValueError("model has no trainable parameters")
    batch_values = _materialize_batches(batches)
    # Accept both a flattened direction and a tuple/list matching model
    # parameters.  The latter is convenient when callers already use the
    # ``torch.autograd.grad`` representation and avoids fragile concatenation
    # at adapter boundaries.
    flat_vector = _coerce_flat_vector(vector, parameters)
    vector_parts = _split_vector(flat_vector, parameters)
    was_training = bool(model.training)
    model.eval()
    try:
        loss = _average_loss(
            model,
            batch_values,
            forward_fn=forward_fn,
            objective=objective,
            label_source=label_source,
            loss_fn=loss_fn,
        )
        first = torch.autograd.grad(loss, parameters, create_graph=True, allow_unused=True)
        dot_terms = []
        for parameter, gradient, direction in zip(parameters, first, vector_parts):
            if gradient is None:
                dot_terms.append((parameter * 0.0).sum())
            else:
                dot_terms.append((gradient * direction).sum())
        dot = torch.stack(dot_terms).sum()
        if not dot.requires_grad:
            return torch.zeros_like(flat_vector)
        second = torch.autograd.grad(dot, parameters, allow_unused=True)
        return _flatten(
            [value if value is not None else torch.zeros_like(parameter) for parameter, value in zip(parameters, second)],
            device=flat_vector.device,
            dtype=flat_vector.dtype,
        ).detach()
    finally:
        model.train(was_training)


# A short alias is useful in notebooks and keeps compatibility with common
# PyTorch curvature terminology.
hessian_hvp = hessian_vector_product
hvp = hessian_vector_product


def _random_direction(parameters: Sequence[Any], *, seed: int, distribution: str = "normal") -> Any:
    torch = _torch()
    generator = torch.Generator(device=parameters[0].device.type).manual_seed(int(seed))
    if distribution == "rademacher":
        chunks = [
            (torch.randint(0, 2, parameter.shape, generator=generator, device=parameter.device, dtype=torch.int64) * 2 - 1).to(parameter.dtype)
            for parameter in parameters
        ]
    else:
        chunks = [torch.randn(parameter.shape, generator=generator, device=parameter.device, dtype=parameter.dtype) for parameter in parameters]
    vector = _flatten(chunks)
    return vector / torch.linalg.vector_norm(vector).clamp_min(EPS)


def hessian_top_eigenvalue_summary(
    model: Any,
    batches: Iterable[Any] | Any,
    *,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    objective: str = "cross_entropy",
    label_source: str = "true",
    loss_fn: Callable[..., Any] | None = None,
    iterations: int = 20,
    tolerance: float = 1e-5,
    seed: int = 20260827,
    vector: Any | None = None,
    max_iter: int | None = None,
    num_iters: int | None = None,
) -> dict[str, Any]:
    """Estimate the dominant Hessian eigenvalue with power iteration.

    ``eigenvalue`` is the Rayleigh quotient of the final iterate.  Standard
    power iteration converges to the eigenvalue of largest magnitude; both the
    signed value and its absolute value are reported explicitly.
    """

    torch = _torch()
    parameters = _trainable_parameters(model)
    if not parameters:
        raise ValueError("model has no trainable parameters")
    if max_iter is not None:
        iterations = max_iter
    if num_iters is not None:
        iterations = num_iters
    iterations = int(iterations)
    tolerance = float(tolerance)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    batches_values = _materialize_batches(batches)
    if vector is None:
        current = _random_direction(parameters, seed=seed)
    else:
        current = _coerce_flat_vector(vector, parameters)
        current = current / torch.linalg.vector_norm(current).clamp_min(EPS)
    previous: float | None = None
    eigenvalue = 0.0
    converged = False
    performed = 0
    for index in range(iterations):
        hv = hessian_vector_product(
            model,
            batches_values,
            current,
            forward_fn=forward_fn,
            objective=objective,
            label_source=label_source,
            loss_fn=loss_fn,
        )
        eigenvalue = float(torch.dot(current, hv).item())
        norm = float(torch.linalg.vector_norm(hv).item())
        performed = index + 1
        if norm <= EPS:
            converged = True
            break
        next_vector = hv / norm
        if previous is not None and abs(eigenvalue - previous) <= tolerance * max(1.0, abs(previous)):
            converged = True
            current = next_vector
            break
        previous = eigenvalue
        current = next_vector
    return {
        "status": "computed",
        "method": "power_iteration",
        "eigenvalue": eigenvalue,
        "absolute_eigenvalue": abs(eigenvalue),
        # Alias used by several curvature toolkits.
        "top_eigenvalue": eigenvalue,
        "iterations": performed,
        "converged": converged,
        "tolerance": tolerance,
        "parameter_count": sum(int(parameter.numel()) for parameter in parameters),
        "data_dependent": True,
    }


def top_hessian_eigenvalue(
    model: Any,
    batches: Iterable[Any] | Any,
    *,
    return_summary: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """Convenience wrapper returning a scalar (or full summary) estimate."""

    summary = hessian_top_eigenvalue_summary(model, batches, **kwargs)
    return summary if return_summary else float(summary["eigenvalue"])


def hessian_top_eigenvalue(*args: Any, **kwargs: Any) -> float | dict[str, Any]:
    """Alias for :func:`top_hessian_eigenvalue`."""

    return top_hessian_eigenvalue(*args, **kwargs)


def power_iteration_top_eigenvalue(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias returning the detailed power-iteration summary."""

    return hessian_top_eigenvalue_summary(*args, **kwargs)


def hutchinson_trace_summary(
    model: Any,
    batches: Iterable[Any] | Any,
    *,
    probes: int = 8,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    objective: str = "cross_entropy",
    label_source: str = "true",
    loss_fn: Callable[..., Any] | None = None,
    seed: int = 20260827,
    num_probes: int | None = None,
    num_samples: int | None = None,
) -> dict[str, Any]:
    """Estimate ``trace(H)`` with Rademacher Hutchinson probes."""

    torch = _torch()
    parameters = _trainable_parameters(model)
    if not parameters:
        raise ValueError("model has no trainable parameters")
    if num_probes is not None:
        probes = num_probes
    if num_samples is not None:
        probes = num_samples
    probes = int(probes)
    if probes < 1:
        raise ValueError("probes must be positive")
    batches_values = _materialize_batches(batches)
    estimates: list[float] = []
    for index in range(probes):
        direction = _random_direction(parameters, seed=int(seed) + index, distribution="rademacher")
        # Hutchinson's estimator is unbiased for an unnormalised random
        # vector.  ``_random_direction`` is unit-normalised, so rescale by the
        # parameter dimension to keep the estimator's expected trace.
        dimension = max(1, direction.numel())
        direction = direction * math.sqrt(dimension)
        hv = hessian_vector_product(
            model,
            batches_values,
            direction,
            forward_fn=forward_fn,
            objective=objective,
            label_source=label_source,
            loss_fn=loss_fn,
        )
        estimates.append(float(torch.dot(direction, hv).item()))
    values = torch.tensor(estimates, dtype=torch.float64)
    return {
        "status": "computed",
        "method": "hutchinson_rademacher",
        "trace": float(values.mean().item()),
        "trace_estimate": float(values.mean().item()),
        "trace_std": float(values.std(unbiased=False).item()) if len(estimates) > 1 else 0.0,
        "probe_count": probes,
        "probe_estimates": estimates,
        "parameter_count": int(values.new_tensor(sum(int(parameter.numel()) for parameter in parameters)).item()),
        "data_dependent": True,
        "seed": int(seed),
    }


def hutchinson_trace(
    model: Any,
    batches: Iterable[Any] | Any,
    *,
    return_summary: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """Convenience wrapper returning a scalar (or full trace summary)."""

    summary = hutchinson_trace_summary(model, batches, **kwargs)
    return summary if return_summary else float(summary["trace"])


def estimate_hessian_trace(*args: Any, **kwargs: Any) -> float | dict[str, Any]:
    """Alias for :func:`hutchinson_trace`."""

    return hutchinson_trace(*args, **kwargs)


def hessian_trace(*args: Any, **kwargs: Any) -> float | dict[str, Any]:
    """Alias for :func:`hutchinson_trace`."""

    return hutchinson_trace(*args, **kwargs)


def exact_hessian_matrix(
    model: Any,
    batches: Iterable[Any] | Any,
    *,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    objective: str = "cross_entropy",
    label_source: str = "true",
    loss_fn: Callable[..., Any] | None = None,
    max_parameters: int = 256,
) -> Any:
    """Materialise an exact Hessian matrix by applying HVP to basis vectors.

    This helper is intended for tiny models and tests.  Larger models should
    use the matrix-free power/Hutchinson estimators.
    """

    torch = _torch()
    parameters = _trainable_parameters(model)
    dimension = sum(int(parameter.numel()) for parameter in parameters)
    if dimension > int(max_parameters):
        raise ValueError(f"exact Hessian requested for {dimension} parameters; limit is {max_parameters}")
    if dimension == 0:
        return torch.empty((0, 0), device=_model_device(model))
    batches_values = _materialize_batches(batches)
    columns = []
    basis = torch.eye(dimension, device=parameters[0].device, dtype=parameters[0].dtype)
    for index in range(dimension):
        columns.append(
            hessian_vector_product(
                model,
                batches_values,
                basis[index],
                forward_fn=forward_fn,
                objective=objective,
                label_source=label_source,
                loss_fn=loss_fn,
            )
        )
    return torch.stack(columns, dim=1)


def _parameter_input_batch(batches: Sequence[Any], max_samples: int) -> Any | None:
    """Concatenate model inputs for the sampled-Jacobian probe."""

    torch = _torch()
    pieces: list[Any] = []
    for batch in batches:
        inputs, _labels = split_batch(batch)
        pieces.append(inputs)
    if not pieces:
        return None
    first = pieces[0]
    if isinstance(first, torch.Tensor):
        return torch.cat([item for item in pieces if isinstance(item, torch.Tensor)], dim=0)[:max_samples]
    if isinstance(first, (tuple, list)) and all(isinstance(item, (tuple, list)) for item in pieces):
        width = len(first)
        merged = []
        for index in range(width):
            values = [item[index] for item in pieces]
            if not all(isinstance(value, torch.Tensor) for value in values):
                return None
            merged.append(torch.cat(values, dim=0)[:max_samples])
        return tuple(merged) if isinstance(first, tuple) else merged
    return None


def _first_nested_tensor(value: Any) -> Any | None:
    """Return the first tensor in a nested input payload.

    EEG adapters commonly pass either a tensor, ``(eeg, eog)`` tuple, or a
    mapping.  The helper is intentionally structural so the diagnostics do
    not need to import a dataset-specific batch type.
    """

    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_nested_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_nested_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _has_sequence_input_axis(batches: Sequence[Any]) -> bool:
    """Whether calibration inputs use the EEG ``[B,T,C,S]`` convention.

    Built-in EEG decoders flatten epochs internally and restore the leading
    ``T`` axis on their bundle representations.  A positive channel feature
    axis therefore shifts by one for sequence inputs (for example ``1`` →
    ``2``), while a feature axis of ``-1`` is unchanged.  This flag only
    informs that explicit representation-spec convention; callers can always
    override an axis through ``MetricConfig.feature_axes``.
    """

    for batch in batches:
        inputs, _labels = split_batch(batch)
        tensor = _first_nested_tensor(inputs)
        if tensor is not None and tensor.ndim == 4:
            return True
    return False


def _gradient_matrix(
    model: Any,
    batches: Sequence[Any],
    *,
    forward_fn: Callable[[Any, Any], Any] | None,
    objective: str,
    label_source: str,
    loss_fn: Callable[..., Any] | None,
) -> Any:
    torch = _torch()
    parameters = _trainable_parameters(model)
    rows = []
    was_training = bool(model.training)
    model.eval()
    try:
        for batch in batches:
            inputs, labels = split_batch(batch)
            device = _model_device(model)
            inputs = _move_to_device(inputs, device)
            labels = _move_to_device(labels, device) if labels is not None else None
            logits = _extract_tensor(_invoke_forward(model, inputs, forward_fn))
            loss = _objective_loss(model, batch, logits, labels, objective=objective, label_source=label_source, loss_fn=loss_fn)
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            rows.append(
                _flatten([gradient if gradient is not None else torch.zeros_like(parameter) for parameter, gradient in zip(parameters, gradients)]).detach().cpu()
            )
    finally:
        model.train(was_training)
    return torch.stack(rows) if rows else torch.empty((0, sum(int(parameter.numel()) for parameter in parameters)))


def _nested_batch_size(value: Any) -> int | None:
    """Return the leading batch size of tensor/nested model inputs."""

    torch = _torch()
    if isinstance(value, torch.Tensor):
        return int(value.shape[0]) if value.ndim > 0 else None
    if isinstance(value, (tuple, list)):
        for item in value:
            result = _nested_batch_size(item)
            if result is not None:
                return result
    if isinstance(value, Mapping):
        for item in value.values():
            result = _nested_batch_size(item)
            if result is not None:
                return result
    return None


def _slice_nested_batch(value: Any, index: int) -> Any:
    """Select one leading-batch item while preserving nested input layout."""

    torch = _torch()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        return value[index : index + 1]
    if isinstance(value, tuple):
        return tuple(_slice_nested_batch(item, index) for item in value)
    if isinstance(value, list):
        return [_slice_nested_batch(item, index) for item in value]
    if isinstance(value, Mapping):
        return {key: _slice_nested_batch(item, index) for key, item in value.items()}
    return value


def empirical_fisher_diagonal(
    model: Any,
    batches: Iterable[Any] | Any,
    *,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    objective: str = "cross_entropy",
    label_source: str = "true",
    loss_fn: Callable[..., Any] | None = None,
    max_samples: int = 32,
) -> Any:
    """Estimate the empirical-Fisher diagonal from per-sample gradients.

    For each calibration sample ``g_i = ∇θ ℓ_i`` is computed and the returned
    vector is ``mean_i(g_i ⊙ g_i)``.  This is an objective- and data-dependent
    diagonal estimator; it is not the full Fisher matrix and should not be
    conflated with the Hessian.  ``max_samples`` bounds the expensive loop.
    """

    torch = _torch()
    parameters = _trainable_parameters(model)
    if not parameters:
        raise ValueError("model has no trainable parameters")
    limit = int(max_samples)
    if limit < 1:
        raise ValueError("max_samples must be positive")
    batch_values = _materialize_batches(batches)
    rows: list[Any] = []
    was_training = bool(model.training)
    model.eval()
    device = _model_device(model)
    try:
        for batch in batch_values:
            inputs, labels = split_batch(batch)
            inputs = _move_to_device(inputs, device)
            labels = _move_to_device(labels, device) if labels is not None else None
            count = _nested_batch_size(inputs)
            if count is None:
                raise TypeError("inputs do not expose a leading batch dimension")
            for index in range(count):
                if len(rows) >= limit:
                    break
                sample_inputs = _slice_nested_batch(inputs, index)
                sample_labels = _slice_nested_batch(labels, index) if labels is not None else None
                sample_batch = (sample_inputs, sample_labels)
                output = _invoke_forward(model, sample_inputs, forward_fn)
                logits = _extract_tensor(output)
                loss = _objective_loss(
                    model,
                    sample_batch,
                    logits,
                    sample_labels,
                    objective=objective,
                    label_source=label_source,
                    loss_fn=loss_fn,
                )
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
                rows.append(
                    _flatten(
                        [gradient if gradient is not None else torch.zeros_like(parameter) for parameter, gradient in zip(parameters, gradients)]
                    ).detach()
                )
            if len(rows) >= limit:
                break
    finally:
        model.train(was_training)
    if not rows:
        return torch.empty(sum(int(parameter.numel()) for parameter in parameters), device=device)
    gradients = torch.stack(rows)
    return gradients.square().mean(dim=0)


def empirical_fisher_summary(
    model: Any,
    batches: Iterable[Any] | Any,
    *,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    objective: str = "cross_entropy",
    label_source: str = "true",
    loss_fn: Callable[..., Any] | None = None,
    max_samples: int = 32,
) -> dict[str, Any]:
    """Return scalar summaries for :func:`empirical_fisher_diagonal`."""

    torch = _torch()
    batch_values = _materialize_batches(batches)
    observed = 0
    for batch in batch_values:
        size = _nested_batch_size(split_batch(batch)[0])
        if size is not None:
            observed += int(size)
    observed = min(int(max_samples), observed)
    diagonal = empirical_fisher_diagonal(
        model,
        batch_values,
        forward_fn=forward_fn,
        objective=objective,
        label_source=label_source,
        loss_fn=loss_fn,
        max_samples=max_samples,
    )
    values = diagonal.detach().float()
    return {
        "status": "computed",
        "method": "per_sample_gradient_square_diagonal",
        "objective": str(objective),
        "label_source": str(label_source),
        "sample_count": observed,
        "parameter_dim": int(values.numel()),
        "trace": float(values.sum().item()),
        "mean": float(values.mean().item()) if values.numel() else 0.0,
        "max": float(values.max().item()) if values.numel() else 0.0,
        "nonzero_fraction": float((values > 0).float().mean().item()) if values.numel() else 0.0,
        "data_dependent": True,
    }


# Familiar aliases used in experiment notebooks.
fisher_diagonal = empirical_fisher_diagonal
fisher_summary = empirical_fisher_summary


def _json_safe(value: Any) -> Any:
    """Convert tensors/scalars recursively for report consumers."""

    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _capture_bundle_representations(
    model: Any,
    batches: Sequence[Any],
    names: Sequence[str],
    *,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    forward_bundle_fn: Callable[[Any, Any], Any] | None = None,
) -> dict[str, Any]:
    """Capture representations exposed by ``forward_bundle`` outputs.

    Built-in EEG decoders return an ``EEGForwardBundle``.  Keeping this path
    separate from module hooks means custom adapters can expose semantic
    stages (``fusion``, ``embedding``, ``classifier_input``) without forcing
    callers to know the internal module tree.
    """

    torch = _torch()
    collected: dict[str, list[Any]] = {str(name): [] for name in names}
    was_training = bool(model.training)
    model.eval()
    device = _model_device(model)
    try:
        with torch.no_grad():
            for batch in batches:
                inputs, _labels = split_batch(batch)
                inputs = _move_to_device(inputs, device)
                if forward_bundle_fn is not None:
                    output = forward_bundle_fn(model, inputs)
                elif hasattr(model, "forward_bundle"):
                    output = model.forward_bundle(inputs)
                elif forward_fn is not None:
                    output = _invoke_forward(model, inputs, forward_fn)
                else:
                    output = _invoke_forward(model, inputs, None)
                representations = _bundle_representations(output)
                if representations is None:
                    continue
                for name in collected:
                    value = representations.get(name)
                    if isinstance(value, torch.Tensor):
                        collected[name].append(value.detach().cpu())
    finally:
        model.train(was_training)
    result: dict[str, Any] = {}
    for name, values in collected.items():
        result[name] = torch.cat(values, dim=0) if values else None
    return result


def _capture_bundle_attention(
    model: Any,
    batches: Sequence[Any],
    *,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    forward_bundle_fn: Callable[[Any, Any], Any] | None = None,
) -> dict[str, Any]:
    """Capture and summarize attention maps exposed by a forward bundle.

    Attention is deliberately kept separate from representation taps because
    not every decoder has attention and because the normalization axis is an
    architectural semantic (BrainUICL uses the head axis, while standard
    MHA uses the key/token axis).  The returned object is JSON-safe and marks
    unavailable attention explicitly instead of manufacturing zero maps.
    """

    torch = _torch()
    from . import lop_metrics as metrics

    collected: dict[str, list[Any]] = {}
    axis: int | None = None
    architecture_metadata: dict[str, Any] = {}
    was_training = bool(model.training)
    model.eval()
    device = _model_device(model)
    try:
        with torch.no_grad():
            for batch in batches:
                inputs, _labels = split_batch(batch)
                inputs = _move_to_device(inputs, device)
                if forward_bundle_fn is not None:
                    output = forward_bundle_fn(model, inputs)
                elif hasattr(model, "forward_bundle"):
                    output = model.forward_bundle(inputs)
                elif forward_fn is not None:
                    output = _invoke_forward(model, inputs, forward_fn)
                else:
                    output = _invoke_forward(model, inputs, None)
                values = getattr(output, "attention", None)
                if isinstance(output, Mapping) and isinstance(output.get("attention"), Mapping):
                    values = output["attention"]
                if isinstance(values, Mapping):
                    for name, value in values.items():
                        if isinstance(value, torch.Tensor):
                            collected.setdefault(str(name), []).append(value.detach().cpu())
                metadata = getattr(output, "metadata", None)
                if isinstance(output, Mapping) and isinstance(output.get("metadata"), Mapping):
                    metadata = output["metadata"]
                if isinstance(metadata, Mapping):
                    if "attention_normalization_axis" in metadata:
                        axis = int(metadata["attention_normalization_axis"])
                    for key in ("architecture", "input_layout", "d_model", "heads", "eeg_channels", "eog_channels"):
                        if key in metadata:
                            architecture_metadata[key] = metadata[key]
    finally:
        model.train(was_training)
    if not collected:
        result: dict[str, Any] = {
            "status": "unavailable",
            "reason": "model did not expose attention maps in its forward bundle",
        }
        if axis is not None:
            result["normalization_axis"] = axis
        if architecture_metadata:
            result["metadata"] = architecture_metadata
        return result
    if axis is None:
        axis = -1
    layers: list[dict[str, Any]] = []
    for name, values in collected.items():
        try:
            merged = torch.cat(values, dim=0)
            layers.append({"layer": name, **metrics.attention_summary(merged, normalization_axis=axis)})
        except Exception as error:
            layers.append({"layer": name, "status": "error", "error": f"{type(error).__name__}: {error}"})
    result = {
        "status": "computed",
        "layers": layers,
        "normalization_axis": int(axis),
        "source": "forward_bundle",
    }
    if architecture_metadata:
        result["metadata"] = architecture_metadata
    return result


def _diagnose_model_impl(
    model: Any,
    batches: Iterable[Any] | Any,
    *,
    config: MetricConfig | None = None,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    forward_bundle_fn: Callable[[Any, Any], Any] | None = None,
    layer_map: Mapping[str, Any] | None = None,
    baseline: Any | Mapping[str, Any] | None = None,
    loss_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run a bounded, architecture-agnostic LoP diagnostic report.

    ``layer_map`` maps names to modules and is optional.  If ``baseline`` is a
    model, CKA/Procrustes are computed on the same calibration batches; a
    mapping may instead provide pre-captured baseline representations.  The
    report separates data-independent parameter norms from explicitly
    data-dependent representation/Jacobian/curvature quantities.
    """

    torch = _torch()
    cfg = config if config is not None else MetricConfig()
    if not isinstance(cfg, MetricConfig):
        cfg = MetricConfig.from_mapping(dict(cfg))
    batch_values = _materialize_batches(batches, cfg.max_batches)
    sequence_input = _has_sequence_input_axis(batch_values)
    # Import the lightweight tensor summaries lazily to keep this module
    # importable in environments without torch.
    from . import lop_metrics as metrics

    report: dict[str, Any] = {
        "schema_version": "edgeforge-lop-diagnostics-v1",
        "protocol": "calibration-based-lop-diagnostics-v1",
        "status": "computed",
        "scientific_conclusion_allowed": False,
        "config": cfg.to_dict(),
        "data": {
            "batch_count": len(batch_values),
            "label_available": any(split_batch(batch)[1] is not None for batch in batch_values),
            "objective": cfg.objective,
            "label_source": cfg.label_source,
            "data_dependent": True,
            "calibration_manifest": calibration_manifest_provenance(
                cfg.calibration_manifest,
                declared_digest=cfg.calibration_manifest_digest,
            ),
        },
        "parameters": metrics.parameter_norm_summary(model, baseline if hasattr(baseline, "named_parameters") else None),
    }
    # Parameter norms/singular values describe the current checkpoint (and an
    # optional reference checkpoint); unlike activation/Jacobian/curvature
    # probes they do not require a calibration batch.
    report["parameters"]["data_dependent"] = False
    try:
        report["parameter_spectra"] = metrics.parameter_spectral_summary(model)
    except Exception as error:
        report["parameter_spectra"] = {"status": "error", "error": f"{type(error).__name__}: {error}"}

    captures: dict[str, Any] = {}
    baseline_captures: dict[str, Any] = {}
    # ``representation_specs`` is the preferred semantic adapter for EEG
    # decoders.  A caller may still provide a module ``layer_map`` for generic
    # PyTorch models; mappings whose values are metadata dictionaries are
    # interpreted as bundle specs rather than hook targets.
    specs: Mapping[str, Any] = layer_map or {}
    if not specs and hasattr(model, "representation_specs"):
        try:
            candidate = model.representation_specs()
            if isinstance(candidate, Mapping):
                specs = candidate
        except Exception:
            specs = {}
    module_targets = bool(specs) and all(hasattr(value, "register_forward_hook") for value in specs.values())
    if cfg.include_representations and specs:
        try:
            if module_targets:
                capture_forward = lambda module, batch: _extract_tensor(_invoke_forward(module, split_batch(batch)[0], forward_fn))
                captures = metrics.capture_representations(model, batch_values, specs, forward_fn=capture_forward)
                if isinstance(baseline, Mapping):
                    baseline_captures = dict(baseline)
                elif baseline is not None and hasattr(baseline, "named_parameters"):
                    baseline_captures = metrics.capture_representations(baseline, batch_values, specs, forward_fn=capture_forward)
            else:
                names = [str(name) for name in specs]
                captures = _capture_bundle_representations(
                    model,
                    batch_values,
                    names,
                    forward_fn=forward_fn,
                    forward_bundle_fn=forward_bundle_fn,
                )
                if isinstance(baseline, Mapping):
                    baseline_captures = dict(baseline)
                elif baseline is not None and hasattr(baseline, "named_parameters"):
                    baseline_captures = _capture_bundle_representations(
                        baseline,
                        batch_values,
                        names,
                        forward_fn=forward_fn,
                        forward_bundle_fn=forward_bundle_fn,
                    )
        except Exception as error:  # diagnostics should not abort an experiment
            report["representations_error"] = f"{type(error).__name__}: {error}"
        representation_report: dict[str, Any] = {}
        for name, values in captures.items():
            if values is None:
                representation_report[name] = None
                continue
            metadata = specs.get(name, {})
            if isinstance(metadata, Mapping):
                hinted_axis = metadata.get("feature_axis", -1)
                hinted_kind = str(metadata.get("kind", "unknown"))
            else:
                hinted_axis, hinted_kind = -1, "unknown"
            axis_was_configured = name in cfg.feature_axes
            axis = cfg.feature_axes.get(name, int(hinted_axis))
            axis_adjusted = False
            # Bundle representations from the built-in EEG decoders are
            # restored to ``[B,T,...]`` for sequence inputs.  Specs with a
            # non-negative channel axis describe the flattened epoch layout
            # (``[B,C,S]``), so prepend-aware sequence restoration shifts that
            # axis by one.  A caller-supplied feature_axes entry is treated as
            # authoritative; ``layout=flattened_epoch_map`` is the explicit
            # escape hatch for taps intentionally kept flattened (for example
            # Conformer's local projection).
            layout = metadata.get("layout") if isinstance(metadata, Mapping) else None
            sequence_policy = metadata.get("sequence_axis_policy") if isinstance(metadata, Mapping) else None
            if (
                sequence_input
                and not axis_was_configured
                and int(axis) >= 0
                and layout != "flattened_epoch_map"
                and sequence_policy in {"prepend", "restored"}
            ):
                axis = int(axis) + 1
                axis_adjusted = True
            item: dict[str, Any] = {
                "spectrum": metrics.spectral_summary(values, feature_axis=axis, max_observations=cfg.max_observations),
                "activation": metrics.activation_summary(values, kind=hinted_kind),
                "feature_axis": axis,
                "kind": hinted_kind,
                "axis_source": "config" if axis_was_configured else "model_spec",
                "sequence_axis_adjusted": axis_adjusted,
            }
            reference = baseline_captures.get(name)
            if reference is not None:
                item["cka_to_reference"] = metrics.linear_cka(values, reference, feature_axis=axis, max_observations=cfg.max_observations)
                item["procrustes_residual_to_reference"] = metrics.procrustes_residual(values, reference, feature_axis=axis, max_observations=cfg.max_observations)
            representation_report[name] = item
        report["representations"] = representation_report
    else:
        report["representations"] = {}

    # Attention maps are optional semantic outputs.  Keep them as a separate
    # top-level section so a model without attention is distinguishable from a
    # model whose attention was not instrumented.  The bundle metadata records
    # the normalization axis (BrainUICL: head axis 1; standard MHA: key axis
    # -1), which is required before comparing entropy across architectures.
    if hasattr(model, "forward_bundle"):
        try:
            report["attention"] = _capture_bundle_attention(
                model,
                batch_values,
                forward_fn=forward_fn,
                forward_bundle_fn=forward_bundle_fn,
            )
        except Exception as error:
            report["attention"] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    else:
        report["attention"] = {"status": "unavailable", "reason": "model has no forward_bundle attention interface"}

    # Jacobian/NTK uses a bounded concatenation of model inputs.  A nested
    # multi-input payload is supported; unsupported custom batches are simply
    # reported as skipped rather than turning the whole run into a failure.
    if cfg.include_jacobian and cfg.jacobian_samples > 0:
        jacobian_inputs = _parameter_input_batch(batch_values, cfg.jacobian_samples)
        if jacobian_inputs is None:
            report["jacobian"] = {"status": "skipped", "reason": "inputs are not tensor-like"}
        else:
            try:
                jacobian = metrics.sampled_parameter_jacobian(
                    model,
                    jacobian_inputs,
                    forward_fn=lambda module, item: _extract_tensor(_invoke_forward(module, item, forward_fn)),
                    max_samples=cfg.jacobian_samples,
                )
                report["jacobian"] = metrics.jacobian_summary(jacobian)
            except Exception as error:
                report["jacobian"] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    else:
        report["jacobian"] = {"status": "disabled"}

    if cfg.include_gradients:
        try:
            gradients = _gradient_matrix(
                model,
                batch_values,
                forward_fn=forward_fn,
                objective=cfg.objective,
                label_source=cfg.label_source,
                loss_fn=loss_fn,
            )
            report["gradients"] = metrics.gradient_summary(gradients)
        except Exception as error:
            report["gradients"] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    else:
        report["gradients"] = {"status": "disabled"}

    if cfg.include_fisher:
        try:
            report["fisher"] = empirical_fisher_summary(
                model,
                batch_values,
                forward_fn=forward_fn,
                objective=cfg.objective,
                label_source=cfg.label_source,
                loss_fn=loss_fn,
                max_samples=cfg.fisher_samples,
            )
        except Exception as error:
            report["fisher"] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    else:
        report["fisher"] = {"status": "disabled"}

    if cfg.include_hessian and cfg.hessian_mode != "none":
        hessian_report: dict[str, Any] = {
            "status": "computed",
            "mode": cfg.hessian_mode,
            "objective": cfg.objective,
            "label_source": cfg.label_source,
            "data_dependent": True,
        }
        try:
            if cfg.hessian_mode in {"hvp", "power", "all"}:
                top_summary = hessian_top_eigenvalue_summary(
                    model,
                    batch_values,
                    forward_fn=forward_fn,
                    objective=cfg.objective,
                    label_source=cfg.label_source,
                    loss_fn=loss_fn,
                    iterations=cfg.hessian_iterations,
                    tolerance=cfg.hessian_tolerance,
                    seed=cfg.seed,
                )
                # Keep the headline metric scalar (matching ``ntk_trace`` in
                # ``jacobian_summary``) while retaining convergence metadata.
                hessian_report["top_eigenvalue"] = top_summary["eigenvalue"]
                hessian_report["top_eigenvalue_abs"] = top_summary["absolute_eigenvalue"]
                hessian_report["top_eigenvalue_summary"] = top_summary
            if cfg.hessian_mode in {"hvp", "hutchinson", "all"}:
                trace_summary = hutchinson_trace_summary(
                    model,
                    batch_values,
                    probes=cfg.hessian_probes,
                    forward_fn=forward_fn,
                    objective=cfg.objective,
                    label_source=cfg.label_source,
                    loss_fn=loss_fn,
                    seed=cfg.seed,
                )
                hessian_report["trace"] = trace_summary["trace"]
                hessian_report["trace_std"] = trace_summary["trace_std"]
                hessian_report["trace_summary"] = trace_summary
            if cfg.hessian_mode in {"exact", "all"}:
                count = sum(int(parameter.numel()) for parameter in _trainable_parameters(model))
                if cfg.exact_hessian_max_params and count <= cfg.exact_hessian_max_params:
                    matrix = exact_hessian_matrix(
                        model,
                        batch_values,
                        forward_fn=forward_fn,
                        objective=cfg.objective,
                        label_source=cfg.label_source,
                        loss_fn=loss_fn,
                        max_parameters=cfg.exact_hessian_max_params,
                    )
                    eigenvalues = torch.linalg.eigvalsh((matrix + matrix.T) / 2.0)
                    hessian_report["exact"] = {
                        "status": "computed",
                        "shape": [int(item) for item in matrix.shape],
                        "trace": float(torch.trace(matrix).item()),
                        "top_eigenvalue": float(eigenvalues.max().item()) if eigenvalues.numel() else 0.0,
                        "min_eigenvalue": float(eigenvalues.min().item()) if eigenvalues.numel() else 0.0,
                    }
                else:
                    hessian_report["exact"] = {"status": "skipped", "reason": "parameter_count_exceeds_limit"}
        except Exception as error:
            hessian_report = {"status": "error", "mode": cfg.hessian_mode, "error": f"{type(error).__name__}: {error}"}
        report["hessian"] = hessian_report
    else:
        report["hessian"] = {"status": "disabled"}

    return _json_safe(report)


def diagnose_model(
    model: Any,
    batches: Iterable[Any] | Any,
    *,
    config: MetricConfig | None = None,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    forward_bundle_fn: Callable[[Any, Any], Any] | None = None,
    layer_map: Mapping[str, Any] | None = None,
    baseline: Any | Mapping[str, Any] | None = None,
    loss_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run diagnostics with an explicit BatchNorm/dropout state policy.

    Curvature and representation values can change solely because BatchNorm
    running statistics or dropout masks changed during a probe.  When
    ``MetricConfig.freeze_batch_norm`` is true, the complete model is put in
    evaluation mode for the duration of this call (thereby freezing BN
    running statistics and disabling dropout).  Every module's original
    ``training`` flag is restored in a ``finally`` block, including after an
    exception.  The policy is recorded in the returned provenance block.
    """

    cfg = config if config is not None else MetricConfig()
    if not isinstance(cfg, MetricConfig):
        cfg = MetricConfig.from_mapping(dict(cfg))
    # Save per-module flags rather than only ``model.training``: callers may
    # intentionally keep a submodule (for example a frozen feature extractor)
    # in a different mode.
    try:
        module_modes = {module: bool(module.training) for module in model.modules()}
    except AttributeError:
        module_modes = {model: bool(getattr(model, "training", False))}
    bn_modules = []
    try:
        torch = _torch()
        bn_modules = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    except (AttributeError, RuntimeError):
        bn_modules = []
    if cfg.freeze_batch_norm:
        model.eval()
    try:
        report = _diagnose_model_impl(
            model,
            batches,
            config=cfg,
            forward_fn=forward_fn,
            forward_bundle_fn=forward_bundle_fn,
            layer_map=layer_map,
            baseline=baseline,
            loss_fn=loss_fn,
        )
        report["batch_norm"] = {
            "freeze_running_stats": bool(cfg.freeze_batch_norm),
            "module_count": len(bn_modules),
            "policy": "eval_during_diagnostic" if cfg.freeze_batch_norm else "caller_mode_with_helper_eval",
        }
        return report
    finally:
        for module, mode in module_modes.items():
            # Assigning the flag directly avoids recursively changing a
            # sibling module's state while restoring mixed caller modes.
            try:
                module.training = mode
            except Exception:
                pass


class LoPDiagnostics:
    """Reusable configured diagnostic runner.

    This thin adapter is convenient for training loops that evaluate several
    checkpoints with the same calibration protocol.  It intentionally keeps
    no model or data state, so one instance can safely be reused across
    subjects; call :meth:`run` with a model and batches for each checkpoint.
    """

    def __init__(
        self,
        config: MetricConfig | Mapping[str, Any] | None = None,
        *,
        forward_fn: Callable[[Any, Any], Any] | None = None,
        forward_bundle_fn: Callable[[Any, Any], Any] | None = None,
        layer_map: Mapping[str, Any] | None = None,
        loss_fn: Callable[..., Any] | None = None,
    ) -> None:
        if config is None:
            self.config = MetricConfig()
        elif isinstance(config, MetricConfig):
            self.config = config
        else:
            self.config = MetricConfig.from_mapping(dict(config))
        self.forward_fn = forward_fn
        self.forward_bundle_fn = forward_bundle_fn
        self.layer_map = layer_map
        self.loss_fn = loss_fn

    def run(
        self,
        model: Any,
        batches: Iterable[Any] | Any,
        *,
        baseline: Any | Mapping[str, Any] | None = None,
        layer_map: Mapping[str, Any] | None = None,
        forward_fn: Callable[[Any, Any], Any] | None = None,
        forward_bundle_fn: Callable[[Any, Any], Any] | None = None,
        loss_fn: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Run :func:`diagnose_model` using configured defaults."""

        return diagnose_model(
            model,
            batches,
            config=self.config,
            forward_fn=self.forward_fn if forward_fn is None else forward_fn,
            forward_bundle_fn=self.forward_bundle_fn if forward_bundle_fn is None else forward_bundle_fn,
            layer_map=self.layer_map if layer_map is None else layer_map,
            baseline=baseline,
            loss_fn=self.loss_fn if loss_fn is None else loss_fn,
        )

    __call__ = run


__all__ = [
    "MetricConfig",
    "LoPMetricConfig",
    "DiagnosticsConfig",
    "calibration_manifest_digest",
    "calibration_manifest_provenance",
    "split_batch",
    "objective_loss",
    "hessian_vector_product",
    "hessian_hvp",
    "hvp",
    "hessian_top_eigenvalue_summary",
    "hessian_top_eigenvalue",
    "top_hessian_eigenvalue",
    "power_iteration_top_eigenvalue",
    "hutchinson_trace_summary",
    "hutchinson_trace",
    "estimate_hessian_trace",
    "hessian_trace",
    "exact_hessian_matrix",
    "empirical_fisher_diagonal",
    "empirical_fisher_summary",
    "fisher_diagonal",
    "fisher_summary",
    "diagnose_model",
    "LoPDiagnostics",
]
