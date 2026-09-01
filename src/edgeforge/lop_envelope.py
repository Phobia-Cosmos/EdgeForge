"""Convert LoP diagnostic reports into the EdgeForge metric envelope.

The numerical probes live in :mod:`edgeforge.lop_metrics` and
:mod:`edgeforge.lop_diagnostics`; this module deliberately has no PyTorch
dependency.  It is the small integration boundary used by command-line
diagnostics and by adapters that need to persist an ``edgeforge-bundle-v1``
result.

The converter keeps the scientific roles explicit:

* ``predictor`` — representation/spectrum quantities (for example effective
  rank) that may be used as a lagged predictor;
* ``outcome`` — the fixed-budget fresh-vs-warm plasticity outcome;
* ``retention`` — old-task evaluation only;
* ``diagnostic`` — Jacobian/NTK, Hessian, Fisher, gradients, attention,
  activation and parameter-state measurements.

The role is metadata, not an interpretation.  In particular, emitting a
``predictor`` row does not make a correlation a causal LoP claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


MAX_METRICS = 20_000
_SKIP_KEYS = {
    "status",
    "reason",
    "error",
    "shape",
    "feature_axis",
    "sequence_axis_adjusted",
    "axis_source",
    "kind",
    "source",
    "representation_level",
    "normalization_axis",
    "protocol",
    "label_source",
    "objective",
    "mode",
    "method",
    "data_dependent",
    "batch_norm_policy",
}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _unit(name: str) -> str:
    lowered = name.lower()
    if any(
        token in lowered
        for token in (
            "acc",
            "mf1",
            "auc",
            "cka",
            "procrustes",
            "gap",
            "gain",
            "fraction",
            "rate",
            "agreement",
            "similarity",
            "cosine",
            "coverage",
            "normalized",
        )
    ):
        return "ratio"
    if "count" in lowered or "samples" in lowered or "observations" in lowered:
        return "count"
    if lowered.endswith("_ms") or "latency" in lowered:
        return "ms"
    return "scalar"


def _role(path: str, explicit: str | None = None) -> str:
    """Infer a conservative metric role from a normalized path."""

    if explicit in {"predictor", "outcome", "retention", "diagnostic"}:
        return explicit
    lowered = path.lower()
    if "retention" in lowered or "forgetting" in lowered:
        return "retention"
    if "fresh_gap" in lowered or "fresh_auc_gap" in lowered or "plasticity" in lowered:
        return "outcome"
    # Only rank-like spectrum values are predictors by default.  Other
    # spectrum entries (condition number, tail energy, singular values) remain
    # mechanism diagnostics and cannot silently become the primary predictor.
    if ("spectrum" in lowered or "spectra" in lowered) and (
        lowered.endswith("effective_rank")
        or lowered.endswith("stable_rank")
        or lowered.endswith("effective_rank_normalized")
        or lowered.endswith("stable_rank_normalized")
    ):
        return "predictor"
    return "diagnostic"


def _join(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _budget_items(items: list[dict[str, Any]], max_metrics: int) -> list[dict[str, Any]]:
    """Apply a deterministic, role-aware metric budget."""

    limit = max(1, int(max_metrics))
    if len(items) <= limit:
        return list(items)
    priority = [
        item
        for item in items
        if item.get("context", {}).get("metric_role") in {"predictor", "outcome", "retention"}
    ]
    diagnostics = [
        item
        for item in items
        if item.get("context", {}).get("metric_role") == "diagnostic"
    ]
    if len(priority) >= limit:
        return priority[:limit]
    return priority + diagnostics[: limit - len(priority)]


class _Collector:
    def __init__(self, *, namespace: str, max_metrics: int) -> None:
        self.namespace = namespace
        self.max_metrics = max(1, int(max_metrics))
        self.items: list[dict[str, Any]] = []
        self.seen: set[tuple[str, int | None, str]] = set()

    def finalize(self) -> list[dict[str, Any]]:
        """Apply the output budget without dropping primary evidence first.

        A run over many architectures can easily exceed the control-plane
        metric limit because parameter spectra contain long singular-value
        vectors.  Predictor/outcome/retention rows are retained before the
        lower-priority mechanism diagnostics; this makes truncation explicit
        and deterministic rather than dependent on architecture order.
        """

        return _budget_items(self.items, self.max_metrics)

    def add(
        self,
        name: str,
        value: Any,
        *,
        step: int | None,
        context: Mapping[str, Any],
        role: str | None = None,
    ) -> None:
        number = _finite(value)
        if number is None:
            return
        clean_name = str(name).strip(".")[:240]
        if not clean_name:
            return
        # Keep a bounded overflow for final role-aware pruning.  Primary rows
        # are always admitted; very large diagnostic vectors are capped before
        # they can exhaust memory.
        overflow_limit = self.max_metrics * 4
        if len(self.items) >= overflow_limit and _role(clean_name, role) == "diagnostic":
            return
        clean_context = {str(key): value for key, value in context.items() if value is not None}
        clean_context["metric_role"] = _role(clean_name, role)
        # A stable JSON representation is unnecessary here; sorting by key and
        # repr keeps unusual scalar metadata (for example tuples) hashable.
        context_key = repr(sorted(clean_context.items(), key=lambda item: item[0]))
        key = (clean_name, step, context_key)
        if key in self.seen:
            return
        self.seen.add(key)
        self.items.append(
            {
                "namespace": self.namespace,
                "name": clean_name,
                "value": number,
                "step": step,
                "unit": _unit(clean_name),
                "context": clean_context,
            }
        )

    def flatten(
        self,
        value: Any,
        prefix: str,
        *,
        step: int | None,
        context: Mapping[str, Any],
        role: str | None = None,
        list_step: bool = True,
    ) -> None:
        """Flatten scalar leaves while retaining explicit curve row steps."""

        if _finite(value) is not None:
            self.add(prefix, value, step=step, context=context, role=role)
            return
        if isinstance(value, Mapping):
            explicit_role = value.get("metric_role") if isinstance(value.get("metric_role"), str) else role
            for key, item in value.items():
                key_text = str(key)
                if key_text in _SKIP_KEYS or key_text == "metric_role":
                    continue
                child_role = explicit_role
                # Curvature/attention/gradient sections are always diagnostic
                # unless the caller explicitly overrides a role.
                if child_role is None and any(
                    token in prefix.lower()
                    for token in ("hessian", "jacobian", "gradient", "fisher", "attention", "parameter", "activation", "linearity")
                ):
                    child_role = "diagnostic"
                self.flatten(
                    item,
                    _join(prefix, key_text),
                    step=step,
                    context=context,
                    role=child_role,
                    list_step=list_step,
                )
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if all(isinstance(item, Mapping) for item in value):
                for index, item in enumerate(value):
                    row_step = item.get("step", step)
                    try:
                        row_step = int(row_step) if row_step is not None else None
                    except (TypeError, ValueError):
                        row_step = step
                    row_context = dict(context)
                    extra = item.get("context")
                    if isinstance(extra, Mapping):
                        row_context.update(extra)
                    self.flatten(
                        item,
                        prefix,
                        step=row_step,
                        context=row_context,
                        role=role,
                        list_step=list_step,
                    )
                return
            # Numeric arrays in a diagnostic report are generally structural
            # (shape or singular-value vectors), not scalar time series.  Only
            # flatten an explicitly requested numeric list as a curve.
            if list_step and all(_finite(item) is not None for item in value):
                for index, item in enumerate(value, start=1):
                    self.add(prefix, item, step=index, context=context, role=role)


def _base_context(
    result: Mapping[str, Any],
    *,
    dataset: Any = None,
    method: Any = None,
    architecture: Any = None,
) -> dict[str, Any]:
    config = result.get("config") if isinstance(result.get("config"), Mapping) else {}
    protocol = result.get("protocol")
    # Some historical BrainUICL results used a full provenance dictionary in
    # ``protocol``.  Keep contexts compact and stable by retaining a scalar
    # protocol/instrumentation identifier; the full dictionary remains in the
    # source JSON and bundle summary.
    if not isinstance(protocol, (str, int, float, bool)):
        protocol = config.get("protocol", result.get("instrumentation"))
    context: dict[str, Any] = {
        "dataset": dataset if dataset is not None else config.get("data", config.get("dataset")),
        "method": method if method is not None else config.get("method", result.get("method")),
        "architecture": architecture,
        "protocol": protocol,
        "objective": config.get("objective"),
        "label_source": config.get("label_source"),
    }
    return {key: value for key, value in context.items() if value is not None}


def _stage_context(base: Mapping[str, Any], stage: Mapping[str, Any], *, stage_id: int) -> dict[str, Any]:
    context = dict(base)
    context["stage"] = int(stage_id)
    for key in ("task", "subject", "split", "target_task", "source_stage"):
        if stage.get(key) is not None:
            context[key] = stage[key]
    return context


def _flatten_stage(
    collector: _Collector,
    stage: Mapping[str, Any],
    *,
    stage_id: int,
    context: Mapping[str, Any],
    prefix_root: str = "task",
) -> None:
    """Collect one generic-run stage or one BrainUICL task row."""

    metrics = stage.get("metrics") if isinstance(stage.get("metrics"), Mapping) else None
    if metrics is not None:
        layers = metrics.get("layers")
        if isinstance(layers, Mapping):
            for layer, item in layers.items():
                if not isinstance(item, Mapping):
                    continue
                layer_name = str(layer)
                spectrum = item.get("spectrum")
                if isinstance(spectrum, Mapping):
                    collector.flatten(
                        spectrum,
                        f"{prefix_root}.spectra.{layer_name}",
                        step=stage_id,
                        context=context,
                        role=None,
                    )
                    # The historical LoP analysis uses transformer_1.  Map
                    # the first transformer-like tap to that spelling while
                    # retaining the native layer name above.
                    if layer_name.lower() in {"transformer", "transformer.0", "encoder.0"}:
                        collector.flatten(
                            spectrum,
                            f"{prefix_root}.spectra.transformer_1",
                            step=stage_id,
                            context=context,
                            role=None,
                        )
                for section, role in (("activation", "diagnostic"), ("drift", "diagnostic")):
                    if isinstance(item.get(section), Mapping):
                        collector.flatten(
                            item[section],
                            f"{prefix_root}.{section}.{layer_name}",
                            step=stage_id,
                            context=context,
                            role=role,
                        )
        for section in ("jacobian", "gradient", "fisher", "hessian", "local_linearity", "attention", "parameters", "parameter_spectra"):
            if section in metrics:
                collector.flatten(
                    metrics[section],
                    f"{prefix_root}.{section}",
                    step=stage_id,
                    context=context,
                    role="diagnostic",
                )
        anchor = metrics.get("anchor_metrics")
        if isinstance(anchor, Mapping):
            collector.flatten(anchor, f"{prefix_root}.anchor", step=stage_id, context=context, role="diagnostic")
    # The generic CLI stores the fixed-anchor report beside ``metrics`` at the
    # stage level.  Keep it in the envelope as a diagnostic namespace so CKA/
    # Procrustes on the same calibration data are not silently discarded.
    anchor_top = stage.get("anchor_metrics")
    if isinstance(anchor_top, Mapping):
        collector.flatten(anchor_top, f"{prefix_root}.anchor", step=stage_id, context=context, role="diagnostic")

    # BrainUICL/native task rows use these top-level sections instead of the
    # generic ``metrics`` wrapper.
    for section, role in (
        ("spectra", None),
        ("spectrum", None),
        ("plasticity", "outcome"),
        ("current_before", "diagnostic"),
        ("current_after", "diagnostic"),
        ("old_generalization_after", "retention"),
        ("pseudo_labels", "diagnostic"),
        ("pseudo_labels_on_clean_current", "diagnostic"),
        ("guiding_cpc_losses", "diagnostic"),
        ("training", "diagnostic"),
        ("importance", "diagnostic"),
        ("noise", "diagnostic"),
        ("defense", "diagnostic"),
        ("attack", "diagnostic"),
        ("retention", "retention"),
        ("forgetting", "retention"),
        ("diagnostics", "diagnostic"),
        ("weight_norms", "diagnostic"),
        ("probe", "outcome"),
    ):
        if section in stage:
            if section in {"spectra", "spectrum"} and isinstance(stage[section], Mapping):
                spectrum_value = stage[section]
                collector.flatten(
                    spectrum_value,
                    f"{prefix_root}.{section}",
                    step=stage_id,
                    context=context,
                    role=role,
                )
                # Native BrainUICL reports historically call this tap
                # ``transformer``.  Emit the canonical alias used by the
                # lagged LoP analyzer without dropping the native spelling.
                if "transformer_1" not in spectrum_value and isinstance(spectrum_value.get("transformer"), Mapping):
                    collector.flatten(
                        spectrum_value["transformer"],
                        f"{prefix_root}.spectra.transformer_1",
                        step=stage_id,
                        context=context,
                        role=role,
                    )
                continue
            collector.flatten(
                stage[section],
                f"{prefix_root}.{section}",
                step=stage_id,
                context=context,
                role=role,
            )

    # Legacy BrainUICL task rows expose before/after performance directly.
    # Derive the same adaptation outcome that ``normalize_raeeg_metrics``
    # emits, while retaining the raw snapshots above as diagnostics.
    before = stage.get("current_before") or stage.get("before")
    after = stage.get("current_after") or stage.get("after")
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        for metric in ("acc", "mf1"):
            left, right = _finite(before.get(metric)), _finite(after.get(metric))
            if left is not None and right is not None:
                collector.add(
                    f"plasticity.{metric}_gain",
                    right - left,
                    step=stage_id,
                    context=context,
                    role="outcome",
                )


def diagnostic_to_metrics(
    result: Mapping[str, Any],
    *,
    dataset: Any = None,
    method: Any = None,
    architecture: Any = None,
    namespace: str = "raeeg.research",
    max_metrics: int = MAX_METRICS,
) -> list[dict[str, Any]]:
    """Return an ``edgeforge-bundle-v1`` compatible metric array.

    The function accepts both the generic EEG CLI shape (``runs``/``stages``/
    ``probes``) and the BrainUICL adapter shape (``tasks``).  Existing direct
    metric arrays are validated and returned as a bounded copy, which makes
    the helper safe to call at an adapter boundary more than once.
    """

    if not isinstance(result, Mapping):
        raise TypeError("diagnostic result must be a mapping")
    direct = result.get("metrics")
    has_trajectory = isinstance(result.get("runs"), list) or isinstance(result.get("tasks"), list)
    if isinstance(direct, list) and all(isinstance(item, Mapping) for item in direct) and (direct or not has_trajectory):
        # Do not mutate caller-owned dictionaries.  Preserve already assigned
        # roles, but fill missing envelope fields conservatively.
        copied: list[dict[str, Any]] = []
        for item in direct:
            value = _finite(item.get("value"))
            name = item.get("name")
            if value is None or not isinstance(name, str) or not name.strip():
                continue
            context = dict(item.get("context") or {}) if isinstance(item.get("context"), Mapping) else {}
            context["metric_role"] = _role(name, context.get("metric_role"))
            copied.append(
                {
                    "namespace": str(item.get("namespace") or namespace),
                    "name": name.strip(".")[:240],
                    "value": value,
                    "step": item.get("step"),
                    "unit": str(item.get("unit") or _unit(name)),
                    "context": context,
                }
            )
        return _budget_items(copied, max_metrics)

    collector = _Collector(namespace=namespace, max_metrics=max_metrics)
    base = _base_context(result, dataset=dataset, method=method, architecture=architecture)

    runs = result.get("runs")
    if isinstance(runs, Sequence) and not isinstance(runs, (str, bytes, bytearray)):
        for run in runs:
            if not isinstance(run, Mapping):
                continue
            run_context = dict(base)
            run_architecture = run.get("architecture", architecture)
            if run_architecture is not None:
                run_context["architecture"] = run_architecture
                run_context.setdefault("method", run_architecture)
            for stage in run.get("stages", []):
                if not isinstance(stage, Mapping):
                    continue
                stage_id = stage.get("stage", stage.get("task", 0))
                try:
                    stage_id = int(stage_id)
                except (TypeError, ValueError):
                    continue
                context = _stage_context(run_context, stage, stage_id=stage_id)
                _flatten_stage(collector, stage, stage_id=stage_id, context=context)
            for probe in run.get("probes", []):
                if not isinstance(probe, Mapping):
                    continue
                source = probe.get("after_stage", probe.get("stage", 0))
                target = probe.get("next_task", probe.get("target_task", source))
                try:
                    source, target = int(source), int(target)
                except (TypeError, ValueError):
                    continue
                context = dict(run_context)
                context.update({"source_stage": source, "target_task": target, "stage": target})
                outcome = probe.get("outcome")
                if isinstance(outcome, Mapping):
                    collector.flatten(outcome, "plasticity", step=target, context=context, role="outcome")
                # Preserve the complete warm/fresh learning curves.  Their
                # step is the optimizer update count; source_stage in context
                # prevents collisions across checkpoint transitions.
                curves = probe.get("curves")
                if isinstance(curves, Mapping):
                    collector.flatten(curves, "task.probe.curves", step=target, context=context, role="outcome")
                for key in ("protocol", "seed", "steps", "freeze_batch_norm"):
                    # These are provenance fields, not scalar research rows.
                    _ = key
        return collector.finalize()

    tasks = result.get("tasks")
    if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes, bytearray)):
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            stage = task.get("stage", task.get("task", 0))
            try:
                stage = int(stage)
            except (TypeError, ValueError):
                continue
            context = _stage_context(base, task, stage_id=stage)
            _flatten_stage(collector, task, stage_id=stage, context=context)
        return collector.finalize()

    # A small fallback for a one-off diagnostic report without a trajectory.
    collector.flatten(result, "diagnostic", step=None, context=base, role="diagnostic")
    return collector.finalize()


# Descriptive aliases used by adapters and notebooks.
diagnostics_to_metrics = diagnostic_to_metrics
to_edgeforge_metrics = diagnostic_to_metrics


def edgeforge_bundle_from_diagnostic(
    result: Mapping[str, Any],
    *,
    experiment_id: str,
    workload: str = "eeg-lop",
    spec: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    source_result: Mapping[str, Any] | None = None,
    **metric_kwargs: Any,
) -> dict[str, Any]:
    """Wrap a diagnostic report in the minimal EdgeForge bundle contract."""

    return {
        "schema_version": 1,
        "experiment_id": str(experiment_id),
        "workload": str(workload),
        "spec": dict(spec or {}),
        "metrics": diagnostic_to_metrics(result, **metric_kwargs),
        "summary": dict(result.get("summary") or {}) if isinstance(result.get("summary"), Mapping) else {},
        "source_result": dict(source_result or {}),
        "environment": dict(environment or {}),
        "scientific_conclusion_allowed": False,
    }


__all__ = [
    "MAX_METRICS",
    "diagnostic_to_metrics",
    "diagnostics_to_metrics",
    "to_edgeforge_metrics",
    "edgeforge_bundle_from_diagnostic",
]
