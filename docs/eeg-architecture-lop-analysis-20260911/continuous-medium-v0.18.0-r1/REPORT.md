# EdgeForge continuous EEG architecture LoP pilot

Run: `continuous-medium-v0.18.0-r1`  
Dataset: `eeg-medium/isruc-v0.18.0`  
Status: completed preliminary continuous experiment; `scientific_conclusion_allowed=false`

## Protocol

The source subjects were trained first, then the warm model was carried through the ordered target stream `2 → 11 → 12 → 13 → 14 → 15 → 16 → 17`. Each target file uses its first 10 epochs for adaptation and its last 10 epochs for held-out evaluation. At every stage, a fresh model with the same architecture was adapted from a new initialization using the same batches and budgets `0/5/10/25/50`; the warm model's final state was passed to the next subject. Retention was evaluated on subjects `5,18,19,20` without optimizer updates.

The run contains 5 architectures × 3 seeds × 8 stages = 120 stage records. The architecture implementation is the registry configuration used in the existing EdgeForge benchmark: `lop_mlp` uses hidden widths `[64,64]`, while the compact BrainUICL uses `d_model=64` and classifier hidden width `32` for CPU feasibility.

## Final-budget fresh gap by stage

`fresh_gap = accuracy_fresh(50) − accuracy_warm(50)`. Values are mean ± population SD over the three seeds; the bracket is a percentile bootstrap interval over the three seed values, included as a descriptive stability check.

| architecture | subject 2 | subject 11 | subject 12 | subject 13 | subject 14 | subject 15 | subject 16 | subject 17 | overall mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lop_mlp` | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| `eegnet` | +0.160 | -0.020 | -0.213 | +0.053 | -0.100 | -0.007 | -0.033 | +0.000 | -0.020 |
| `tcn` | +0.007 | -0.080 | +0.147 | +0.067 | -0.013 | +0.000 | +0.087 | -0.273 | -0.008 |
| `transformer` | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| `brainuicl` | +0.013 | -0.133 | +0.167 | -0.293 | -0.187 | +0.007 | +0.093 | -0.080 | -0.052 |

Across all 24 stage/seed records, positive/zero/negative counts were: `lop_mlp 0/24/0`, `eegnet 7/6/11`, `tcn 11/6/7`, `transformer 0/24/0`, `brainuicl 9/0/15`.

## Reading the result

- No architecture satisfies the LoP requirement direction across every stage and seed. The observed gaps change sign by subject for EEGNet, TCN and BrainUICL; LoP-MLP and Transformer are exactly neutral at the final budget in this implementation.
- TCN is positive on subjects 12, 13 and 16 but strongly negative on subject 17, showing why a continuous stream is needed instead of one pooled target split.
- BrainUICL has positive fresh gaps on subjects 12 and 16, but negative gaps on subjects 11, 13, 14 and 17. Its overall mean is negative despite several large positive stages.
- AULC gaps do not rescue the result: overall means were `lop_mlp -0.0367`, `eegnet -0.0117`, `tcn +0.0039`, `transformer -0.0108`, and `brainuicl -0.0104`.

## Limitations

This is a preliminary development experiment. The medium release is explicitly diversity-selected, not a complete ISRUC cohort. The runner uses a compact CPU configuration and a short source pretraining budget, and it does not yet emit a native EdgeForge trajectory catalog with layer-wise ER predictors, generalization gaps, Type-1/Type-2 joint criteria, or full BWT. Consequently the report is evidence that the continuous fresh/warm protocol runs and produces auditable stage-level outcomes, not evidence of a scientific LoP conclusion.

Raw data and outputs remain outside Git under `/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/`.
