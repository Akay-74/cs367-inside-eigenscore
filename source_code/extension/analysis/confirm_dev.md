# Confirmatory results (DEV DRY RUN on CoQA, not a confirmation)

Hypotheses frozen in PLAN_CONFIRMATORY.md (H1-H3: commit 4a26da7; H4: addendum) before these data were analysed.

| dataset | n | accuracy | ES | LexSim | LNE | HYB | **TRI** | **CTC** | **PMF** | tied pairs | within-tie a |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coqa | 1000 | 81.4% | 80.1 | 85.3 | 77.8 | 82.0 | **84.6** | **85.7** | **87.1** | 1.9% | 0.731 |

| test | dataset | ΔAUROC [95% CI] | one-sided p | Holm p | holds? |
|---|---|---|---|---|---|
| H1 LexSim > ES | coqa | +5.1 [+2.8, +7.5] | 0.0000 | 0.0000 | yes |
| H2a TRI > ES | coqa | +4.5 [+2.7, +6.2] | 0.0000 | 0.0000 | yes |
| H3 HYB > ES | coqa | +1.9 [+0.0, +3.8] | 0.0249 | 0.0249 | yes |
| H4a CTC > LexSim | coqa | +0.4 [+0.1, +0.9] | 0.0024 | 0.0048 | yes |
| H4b CTC > ES | coqa | +5.6 [+3.3, +7.9] | 0.0000 | 0.0000 | yes |
| H5a PMF > LexSim | coqa | +1.8 [+0.8, +2.8] | 0.0001 | 0.0003 | yes |
| H5b PMF > ES | coqa | +7.0 [+4.4, +9.5] | 0.0000 | 0.0000 | yes |
| H2b TRI ≥ LexSim − 1pt | coqa | -0.7 [-2.2, +0.8] | – | – | no |

## Verdict

- **H1 LexSim > ES**: confirmed on all datasets
- **H2a TRI > ES**: confirmed on all datasets
- **H3 HYB > ES**: confirmed on all datasets
- **H4a CTC > LexSim**: confirmed on all datasets
- **H4b CTC > ES**: confirmed on all datasets
- **H5a PMF > LexSim**: confirmed on all datasets
- **H5b PMF > ES**: confirmed on all datasets
- **H2b TRI non-inferior to LexSim**: not confirmed
