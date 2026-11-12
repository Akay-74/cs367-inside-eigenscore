# Exploratory hypothesis: cross-depth stability of answer geometry (XDS)

Pre-registered on 2026-11-12, while the full runs were still in progress. It is committed
to git before any AUROC on the full runs was computed. This is an **exploratory** addition:
it does not replace or change the primary comparison in PLAN.md (MLES vs EigenScore).

**Disclosure.** Before this was written, two things had been looked at: label-free
embedding statistics from the 5-question smoke test, and one per-layer single-layer
AUROC chart from 51 partial CoQA questions, produced while testing the analysis script.
No XDS value has been computed on labelled data. The first formulation, linear CKA on the
K x K Gram matrices, was replaced by the RSA formulation below after a *synthetic*
self-test showed that CKA is dominated by each answer's self-similarity and barely reacts
to reshuffling the answers. That test used random data, no labels.

## Motivation and novelty

- **What existing methods read.** EigenScore reads one layer and measures how *spread out*
  the K sampled answers are (a log-volume). Existing cross-layer detectors either track
  one response's hidden-state dynamics with a trained probe (ICR Probe, ACL 2025) or
  train probes over many layers (CLAP, Lyapunov probes, LayerMix). LLM-Check (NeurIPS
  2024) reports that detection quality oscillates across layers, but does not use that as
  a signal.
- **The gap.** We found no prior work that tracks how the *arrangement* of the K sampled
  answers (which answers resemble which) evolves through depth, training-free.
- **Hypothesis.** When the model knows the answer, its samples differ in a consistent way
  through the middle layers (paraphrases stay paraphrases), so the arrangement is stable.
  When it confabulates, the samples are not anchored to a shared fact, and their
  arrangement is re-formed from layer to layer.

## Definitions (fixed)

Notation and band are as in PLAN.md: band = hidden indices 8..24 for Llama-2-7B.
Embeddings are feature-centred exactly as in EigenScore.

- **X1, XDS (cross-depth instability).**
  - For each layer l in the band, D_l is the vector of the K(K-1)/2 = 45 pairwise
    cosine distances between the K answers.
  - XDS = 1 - mean over l = 8..23 of Pearson corr(D_l, D_{l+1}).
  - If all K answers coincide at a layer, that layer pair has corr = 1.
  - Higher XDS = more likely hallucinated.
- **X2, depth slope.** The least-squares slope of the single-layer EigenScore ES(l)
  against l over the band. Pre-registered direction: higher slope (disagreement that
  grows with depth) = more likely hallucinated.
- **X3, depth-aware EigenScore.** (rank(MLES) + rank(XDS)) / 2. Equal weights, no tuning.

## Tests (all secondary / exploratory)

- AUROC of X1, X2 and X3 per dataset, with the same labels as PLAN.md.
- Paired bootstrap (10,000 resamples) vs original EigenScore, and X3 vs MLES.
- **Complementarity.** Spearman rho(MLES, XDS) per dataset. The concept is only
  interesting if XDS is not just a re-encoding of spread: rho clearly below 1 together
  with X3 > MLES.
- **Success criterion, for reporting:** X3 beats MLES with the 95% CI above 0 on at least
  two of the three datasets. Anything less is reported as a negative or inconclusive
  result.
- **Confirmation.** Any positive exploratory finding needs confirmation on fresh
  questions (new seed / new subset) before it is claimed as a result in a paper.
