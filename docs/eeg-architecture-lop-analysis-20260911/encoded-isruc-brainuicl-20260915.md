# BrainUICL LoP diagnostics

- Protocol: `raeeg-lop-posthoc-v2-edgeforge-shared`
- Dataset/method: `ISRUC` / `finetune`
- Stages: `[0, 10, 25]`; seed: `4321`
- Fresh comparator: `random`
- Primary outcome: `fresh_gap_final = accuracy_fresh - accuracy_warm`
- Calibration manifest: `not-provided`; SHA-256: `—`
- Split manifest: `present`; SHA-256: `5344651092da22c1fa3dc068064e6c3c8f5ef9e87178fe8f77f2fb89d67d5346`
- Labels are used only by the held-out/oracle diagnostics; the native unsupervised trainer is unchanged.
- Retention subjects: `[]` (eval-only)

## Summary

- Mean warm accuracy gain: `-0.033333`
- Mean final fresh gap: `-0.433333`
- Mean fresh AULC gap: `-0.291667`
- Mean Transformer effective rank: `9.530901`
- Rank/fresh-gap Pearson: `-0.8080814152806102`

## Stage Results

| stage | target subject | Transformer ER | normalized ER | final fresh gap | fresh AULC gap | old ACC | old MF1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 64 | 10.072767 | 0.503638 | -0.800000 | -0.325000 | — | — |
| 10 | 91 | 10.438911 | 0.521946 | -0.450000 | -0.537500 | — | — |
| 25 | 10 | 8.081023 | 0.404051 | -0.050000 | -0.012500 | — | — |

## Interpretation Boundary

A positive fresh gap is a LoP candidate signal only when the same target split, update budget, optimizer, and seed protocol are repeated across stages and at least three seeds. Rank, CKA/Procrustes, Jacobian/NTK, gradients, activations, and attention are mechanism diagnostics, not standalone LoP evidence.
