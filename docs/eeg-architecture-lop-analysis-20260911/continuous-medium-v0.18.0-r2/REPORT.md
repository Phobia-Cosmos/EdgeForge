# EdgeForge continuous EEG architecture LoP pilot

Run: `continuous-medium-v0.18.0-r2`  
Dataset: `eeg-medium/isruc-v0.18.0`  
Status: completed reproducibility rerun; `scientific_conclusion_allowed=false`

## Protocol

The source subjects were trained first, then the warm model was carried through the ordered target stream `2 → 11 → 12 → 13 → 14 → 15 → 16 → 17`. Each target file uses its first 10 epochs for adaptation and its last 10 epochs for held-out evaluation. At every stage, a fresh model with the same architecture was adapted from a new initialization using the same target batches and budgets `0/5/10/25/50`; the warm model's final state was passed to the next subject. Retention was evaluated on subjects `5,18,19,20` without optimizer updates.

This rerun contains 5 architectures × 3 seeds × 8 stages = 120 stage records and 600 budget-level records. The source optimizer is created once per architecture/seed and reused for all source epochs. The compact CPU configuration uses `lop_mlp` hidden widths `[64,64]` and BrainUICL `d_model=64`, classifier hidden width `32`.

## Final-budget fresh gap by stage

`fresh_gap = accuracy_fresh(50) − accuracy_warm(50)`. Values are mean ± population SD over the three seeds; these are descriptive stability summaries, not inferential confidence intervals.

| architecture | subject 2 | subject 11 | subject 12 | subject 13 | subject 14 | subject 15 | subject 16 | subject 17 | overall mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lop_mlp` | +0.000±0.000 | +0.000±0.000 | +0.000±0.000 | +0.000±0.000 | +0.000±0.000 | +0.000±0.000 | +0.000±0.000 | +0.000±0.000 | +0.000 |
| `eegnet` | +0.160±0.113 | -0.020±0.028 | -0.213±0.132 | +0.053±0.052 | -0.100±0.130 | -0.007±0.075 | -0.033±0.041 | +0.000±0.000 | -0.020 |
| `tcn` | +0.007±0.009 | -0.080±0.113 | +0.147±0.075 | +0.067±0.047 | -0.013±0.025 | +0.000±0.114 | +0.087±0.009 | -0.273±0.075 | -0.008 |
| `transformer` | +0.000±0.000 | +0.000±0.000 | +0.000±0.000 | +0.000±0.000 | +0.000±0.000 | +0.000±0.000 | +0.000±0.000 | +0.000±0.000 | +0.000 |
| `brainuicl` | +0.013±0.226 | -0.133±0.068 | +0.167±0.123 | -0.293±0.116 | -0.187±0.123 | +0.007±0.152 | +0.093±0.057 | -0.080±0.043 | -0.052 |

Across all 24 stage/seed records, positive/zero/negative final gaps were: `lop_mlp 0/24/0`, `eegnet 7/6/11`, `tcn 11/6/7`, `transformer 0/24/0`, and `brainuicl 9/0/15`.

## Aggregate diagnostics

| architecture | mean final gap | mean all-budget gap | mean AULC gap |
| --- | ---: | ---: | ---: |
| `lop_mlp` | +0.0000 | -0.0507 | -0.0367 |
| `eegnet` | -0.0200 | -0.0292 | -0.0117 |
| `tcn` | -0.0075 | -0.0092 | +0.0039 |
| `transformer` | +0.0000 | -0.0300 | -0.0108 |
| `brainuicl` | -0.0517 | +0.0033 | -0.0104 |

Retention is an old-task stability diagnostic only. It is evaluated after each warm stage on the fixed retention subjects and does not enter the adaptation optimizer; it is not a complete BWT estimate.

## Standard EdgeForge gate integration

The continuous summary was converted to the standard EdgeForge trajectory catalog at `audit-v1/trajectory-catalog.json`. Each architecture is represented by three independent seed trajectories with eight explicit transitions (`0->1` through `7->8`); the final-budget fresh-gap rows use `metric_role=outcome`, while 144 retention rows per architecture are retained as a separate inventory.

| architecture | gate status | design | seeds | transitions | direction-supported transitions |
| --- | --- | --- | ---: | ---: | ---: |
| `lop_mlp` | `insufficient-direction` | consistent | 3 | 8 | 0/8 |
| `eegnet` | `blocked-inconsistent-direction` | consistent | 3 | 8 | 0/8 |
| `tcn` | `blocked-inconsistent-direction` | consistent | 3 | 8 | 2/8 |
| `transformer` | `insufficient-direction` | consistent | 3 | 8 | 0/8 |
| `brainuicl` | `blocked-inconsistent-direction` | consistent | 3 | 8 | 2/8 |

The one-command audit output is [audit-summary.json](audit-v1/audit-summary.json), with per-architecture JSON and Markdown gate reports under `audit-v1/gate/`. The gate exit code is non-zero for these results because no architecture passes the strict positive fresh-gap direction requirement; this is the expected evidence status, not a runner failure.

## Budget sweep

The same catalog was audited at every positive budget. The table reports the mean fresh-gap over all 24 stage/seed records; positive/zero/negative counts are also over those 24 records. A positive aggregate mean is not sufficient for the gate, which still requires every transition and seed to be strictly positive with a positive bootstrap lower bound.

| budget | `lop_mlp` | `eegnet` | `tcn` | `transformer` | `brainuicl` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | -0.0458 (6/9/9) | -0.0525 (7/4/13) | -0.0525 (11/4/9) | -0.0217 (8/8/8) | +0.0875 (15/3/6) |
| 10 | -0.0517 (3/13/8) | -0.0300 (7/4/13) | +0.0200 (13/6/5) | -0.0242 (5/15/4) | +0.0408 (13/6/5) |
| 25 | -0.0400 (0/20/4) | +0.0192 (10/5/9) | +0.0208 (15/2/7) | +0.0042 (1/23/0) | -0.0325 (12/0/12) |
| 50 | +0.0000 (0/24/0) | -0.0200 (7/6/11) | -0.0075 (11/6/7) | +0.0000 (0/24/0) | -0.0517 (9/0/15) |

Each parenthesized triplet is `positive/zero/negative`. BrainUICL has the strongest aggregate gap at budget 5, while TCN is slightly positive at budgets 10 and 25; neither remains positive across all transitions and seeds. The budget sweep therefore reinforces the need to report the complete stage-wise curve rather than selecting a favorable budget after the fact.

## Data characteristics linked to LoP diagnostics

The read-only data profile is [DATA_PROFILE.md](data-analysis-v1/DATA_PROFILE.md), with machine-readable output in `data-profile.json`. All 20 subjects contain 100 epochs, so each target stage contributes only 50 adaptation epochs and 50 held-out epochs. Signal RMS forms two broad groups around `1.5–2.7e-6` and `1.5–2.1e-5`; source subject 21 is an additional high-amplitude outlier at `6.07e-5`. Thus the stream contains a large scale/domain shift before any model adaptation is considered.

The combined source label counts are `[161,211,234,79,115]`, giving a majority-class baseline of `0.2925`. Most source runs have accuracy and macro-F1 essentially identical to a constant-class predictor, which indicates underfitting or input-scale/preprocessing mismatch in this pilot. Target class entropy ranges from `1.734` to `2.262` bits (maximum for five classes is about `2.322`), and several target subjects have a class absent from either the adaptation or held-out half. These properties make a stable representation-level LoP signal difficult to separate from class-prior and scale effects.

## EEG drift visualization

The read-only visualization run is archived at `visualizations-v1/`. It generated raw/per-epoch-standardized waveforms, channel-averaged mean-per-epoch Welch spectra, an epoch-level RMS heatmap, feature-space PCA, and adaptation/evaluation label-prior heatmaps. The reproducible command and figure definitions are in [docs/eeg-continuous-lop.md](/home/undefined/Desktop/EdgeForge/docs/eeg-continuous-lop.md) and [visualization README](visualizations-v1/README.md).

The quantitative summary reports a target-subject RMS maximum/minimum ratio of `11.29×` (a `1.053` log10 span). Subjects `11`, `13`, `14` and `17` form the high-amplitude group (`1.48–1.74e-5`), while subjects `2`, `12`, `15` and `16` are around `1.54–2.12e-6`. The normalized spectral Jensen–Shannon distance to the target median is largest for subject `11` (`0.0472`); the other target subjects are between `0.0039` and `0.0124`. Delta power is dominant for every target (`0.512–0.805` of normalized 0.5–40 Hz power), while theta is relatively elevated for subjects `11`, `16` and `17`. PCA of epoch-level log-RMS and band-power features has PC1 explained variance `0.9358`, so the largest visible separation is primarily amplitude/power scale rather than a clean class or task axis.

These plots answer “what does EEG drift look like?”: it appears as subject-specific amplitude bands, occasional within-stream RMS excursions, spectral-profile differences and adaptation/evaluation class-prior changes. They do not answer “did LoP occur?” A subject cluster or a large raw-amplitude shift can produce poor transfer without a warm-vs-fresh fixed-budget gap. LoP still requires the equal-budget held-out comparison and its multi-seed gate.

At budget 50, exploratory correlations show a negative association between signal distance/RMS and fresh-gap for BrainUICL (Pearson about `-0.60/-0.63`) and TCN (about `-0.53/-0.51`), while other architectures and budgets change direction. These are small-sample descriptive associations over 24 stage-seed rows, not causal evidence.

## Interpretation

No architecture satisfies a positive fresh-gap direction across every subject transition and seed. EEGNet, TCN and BrainUICL change sign across subjects; TCN is positive on subjects 12, 13 and 16 but strongly negative on subject 17. LoP-MLP and Transformer are exactly neutral at the final budget in this compact implementation. The AULC diagnostic also does not produce a consistent positive ordering.

The rerun is numerically identical to `continuous-medium-v0.18.0-r1` for source metrics and all stage curves; only wall-clock timing and output metadata differ. This confirms deterministic execution under the recorded seeds and protocol.

## Limitations

The medium release is a diversity-selected development subset, not a complete ISRUC cohort. The runner uses a compact CPU configuration and short source pretraining, and it does not yet emit a native EdgeForge trajectory catalog with layer-wise effective-rank predictors, generalization gaps, Type-1/Type-2 joint criteria, or full BWT. Therefore this report demonstrates an auditable continuous fresh/warm experiment, not a formal scientific LoP conclusion.

Raw data and outputs remain outside Git under `/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/`.
