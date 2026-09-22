# RA-EEG LoP matrix runner v1

`scripts/run-raeeg-lop-matrix.py` is the local orchestration entry point for the next LoP experiments. It expands a compact manifest into independent `dataset × method × condition × subject × checkpoint-stage` cells, runs the read-only BrainUICL instrumentation and the supervised-oracle fixed-budget probe, and stores one `edgeforge-bundle-v1` result per cell.

The runner is deliberately local in this development line. It does not submit tasks to the control plane, copy EEG/checkpoint payloads, modify BrainUICL, or assume multiple GPUs. `--device cuda:0` uses the one visible local GPU; `--device cpu` is a valid fallback. A future scheduler may partition the same explicit cell plan, but must preserve the cell identity and protocol fields.

## Minimal manifest

The compact form is preferable for a method/stage grid:

```json
{
  "schema_version": 1,
  "matrix_id": "isruc-lop-v014-local",
  "version": "0.15.0",
  "brainuicl_root": "/home/undefined/Desktop/bci/code/tta_security/BrainUICL",
  "output_root": "/home/undefined/Disk/ai-storage/EdgeForge/raeeg-lop-matrix/v0.15.0-local",
  "datasets": [{
    "name": "ISRUC",
    "data_root": "/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32",
    "subjects": [1],
    "seeds": [4321],
    "stages": [0, 10, 25, 49],
    "conditions": [{"name": "clean"}],
    "methods": [{
      "name": "finetune",
      "checkpoint_roots": {
        "0": "/home/undefined/Disk/ai-storage/BrainUICL/model_parameter/ISRUC/Pretrain",
        "10": "/path/to/finetune/checkpoints/individual_10",
        "25": "/path/to/finetune/checkpoints/individual_25",
        "49": "/path/to/finetune/checkpoints/individual_49"
      },
      "baseline_checkpoint_root": "/home/undefined/Disk/ai-storage/BrainUICL/model_parameter/ISRUC/Pretrain",
      "fresh_checkpoint_root": "/home/undefined/Disk/ai-storage/BrainUICL/model_parameter/ISRUC/Pretrain"
    }]
  }],
  "defaults": {
    "seed": 4321,
    "device": "cuda:0",
    "max_files": 4,
    "max_batches": 2,
    "importance_batches": 2,
    "probe_steps": "0,10,25,50",
    "train_fraction": 0.5,
    "lr": 1e-5
  }
}
```

`checkpoint_root_template` may be used instead of `checkpoint_roots`; it receives `{dataset}`, `{method}`, `{condition}`, `{subject}`, `{seed}` and `{stage}`. A condition can override `data_root` and may point to a generated shift `manifest`. The canonical dataset remains the clean condition; generated shifts must live in a disjoint output root.

`seeds` is an optional dataset-level array. Omitting it uses the single `defaults.seed` value; specifying `[4321, 4322, 4323]` expands independent seed cells and lets dry-run preflight expose which checkpoint files are still missing. A seed cell is never synthesized from another seed's checkpoint.

Run a plan without executing model code:

```bash
PYTHONPATH=src /home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/run-raeeg-lop-matrix.py \
  --manifest config/raeeg-lop-matrix-v14-local-smoke.json --dry-run
```

The dry-run now performs a read-only input preflight and reports `ready_count`, `blocked_count` and per-cell missing data/checkpoint files. This is the recommended way to check whether a requested seed/stage actually exists before starting GPU work; a blocked preflight is retained in `dry-run.json` and does not fabricate a result.

Run the local matrix and retain a versioned JSONL event log:

```bash
PYTHONPATH=src /home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/run-raeeg-lop-matrix.py \
  --manifest config/raeeg-lop-matrix-v14-local-smoke.json \
  --log-root logs
```

The run directory contains `matrix-plan.json`, `cells/*/{instrumentation,probe,bundle,status}.*`, a cell-level `catalog.json`, a stage-complete `trajectory-catalog.json`, `matrix-report.{json,md}`, `audit.json` and `matrix-summary.json`. When clean and non-clean conditions share the same method/subject/seed/stage, `matrix-report` also includes paired `shift_minus_clean` descriptive deltas. Each command has separate `*.command.json`, `*.stdout.log` and `*.stderr.log` files. Re-running the same manifest resumes only cells whose successful bundle and configuration digest still match; a different manifest is rejected rather than mixed into an existing run directory.

Add `--with-unlabeled-diagnostics` to run `brainuicl-unlabeled-diagnostics.py` for every cell. Its metrics are merged with `metric_role=diagnostic`; it never supplies the LoP outcome and does not read labels or update parameters. This option is useful when the same checkpoint/shift matrix must expose both online-observable confidence and offline supervised-oracle plasticity.

### Optional old-task retention set

A cell may explicitly add a labelled retention set to the method (or dataset/default) configuration:

```json
"retention": {
  "data_root": "/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32",
  "subjects": [1],
  "max_files": 2,
  "batch_size": 1
}
```

The fixed-budget probe updates parameters only with the target cell's train batches. It evaluates the retention batches at step 0 and after each requested step, without labels entering the optimizer and without any retention optimizer update. The resulting `task.forgetting.*` and `task.probe.retention.*` rows are merged with `metric_role=retention`. They are old-task stability diagnostics, not the LoP outcome and not a complete BWT calculation. `--dry-run` checks each retention subject's `data/*.npy` and matching `label/*.npy`; a missing file blocks the cell.

## Evidence interpretation

The instrumentation predictor is `task.spectra.transformer_1.effective_rank` (with the historical `transformer` alias). The probe outcome is `plasticity.acc_gain` and its fixed-budget fresh gap. Optional retention rows have a separate role; `metric_role`, `condition` and `matrix_id` are attached to metric context for auditability; seed remains in the experiment specification so independent seeds can be clustered correctly. The generated audit is intentionally descriptive and always sets `scientific_conclusion_allowed=false`. At least three distinct real seeds, matching stage grids and clean/fresh/equal-budget controls are required before a LoP claim can be considered.

The current local smoke uses one ISRUC subject and one seed only. It verifies the orchestration and bundle contract; its stage-0/stage-10 values must not be reported as evidence that LoP occurs.

The same runner can pair clean data with a controlled derivative. For example, `config/raeeg-lop-matrix-v14-isruc-noise-local.json` compares four stages of clean ISRUC subject 2 with a label-preserving relative-noise 0.5 derivative. In the 0.14.0 single-seed smoke, noise changed the observed effective-rank trajectory and probe gains, but both conditions remained `insufficient-seeds`; a shift-induced performance or spectrum change is not itself LoP. A formal shift experiment must keep the stage order, fresh control, probe budget, labels, and real-seed set identical across conditions.

For the current 0.15.0 retention contract, use `config/raeeg-lop-matrix-v15-retention-local-smoke.json` as a one-cell example and `config/raeeg-lop-matrix-v15-isruc-multiseed-retention-preflight.json` before requesting a real multi-seed run. The preflight currently reports 4 ready seed-4321 stages and 8 blocked seed-4322/4323 stages because those parameter files do not exist. It is an evidence gate, not a synthetic fallback.
