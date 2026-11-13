# Confirmatory results

Hypotheses frozen in PLAN_CONFIRMATORY.md (H1-H3: commit 4a26da7; H4: 47c60ec; H5: 3fcceb6) before these data were analysed.

| dataset | n | accuracy | ES | LexSim | LNE | HYB | **TRI** | **CTC** | **PMF** | tied pairs | within-tie a |
|---|---|---|---|---|---|---|---|---|---|---|---|
| triviaqa | 1000 | 69.9% | 80.5 | 82.5 | 80.6 | 82.0 | **83.2** | **83.7** | **83.7** | 5.9% | 0.699 |
| nq | 1000 | 26.9% | 76.7 | 76.6 | 76.0 | 78.3 | **79.5** | **77.1** | **76.6** | 3.1% | 0.651 |

| test | dataset | ΔAUROC [95% CI] | one-sided p | Holm p | holds? |
|---|---|---|---|---|---|
| H1 LexSim > ES | triviaqa | +2.1 [+0.6, +3.5] | 0.0025 | 0.0175 | yes |
| H2a TRI > ES | triviaqa | +2.7 [+1.7, +3.8] | 0.0000 | 0.0000 | yes |
| H3 HYB > ES | triviaqa | +1.5 [+0.3, +2.7] | 0.0064 | 0.0384 | yes |
| H4a CTC > LexSim | triviaqa | +1.2 [+0.6, +1.8] | 0.0000 | 0.0000 | yes |
| H4b CTC > ES | triviaqa | +3.2 [+1.7, +4.8] | 0.0000 | 0.0000 | yes |
| H5a PMF > LexSim | triviaqa | +1.1 [+0.6, +1.8] | 0.0000 | 0.0000 | yes |
| H5b PMF > ES | triviaqa | +3.2 [+1.7, +4.7] | 0.0000 | 0.0000 | yes |
| H2b TRI ≥ LexSim − 1pt | triviaqa | +0.7 [-0.4, +1.8] | – | – | yes |
| H1 LexSim > ES | nq | -0.1 [-2.4, +2.2] | 0.5400 | 1.0000 | no |
| H2a TRI > ES | nq | +2.7 [+1.3, +4.2] | 0.0000 | 0.0000 | yes |
| H3 HYB > ES | nq | +1.5 [+0.0, +3.1] | 0.0248 | 0.1240 | no |
| H4a CTC > LexSim | nq | +0.5 [+0.2, +0.7] | 0.0001 | 0.0008 | yes |
| H4b CTC > ES | nq | +0.3 [-1.9, +2.6] | 0.3863 | 1.0000 | no |
| H5a PMF > LexSim | nq | +0.0 [-0.9, +1.0] | 0.4661 | 1.0000 | no |
| H5b PMF > ES | nq | -0.1 [-2.5, +2.3] | 0.5305 | 1.0000 | no |
| H2b TRI ≥ LexSim − 1pt | nq | +2.9 [+1.3, +4.5] | – | – | yes (strictly better) |

## Verdict

- **H1 LexSim > ES**: partial
- **H2a TRI > ES**: confirmed on all datasets
- **H3 HYB > ES**: partial
- **H4a CTC > LexSim**: confirmed on all datasets
- **H4b CTC > ES**: partial
- **H5a PMF > LexSim**: partial
- **H5b PMF > ES**: partial
- **H2b TRI non-inferior to LexSim**: confirmed on all datasets
