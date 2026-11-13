# Confirmatory hypotheses (frozen on CoQA, tested on held-out TriviaQA and NQ)

Committed on 2026-11-12, after the CoQA analysis and **before any TriviaQA or NQ result was
computed or looked at**. CoQA served as the development set: the hypotheses below were
chosen after looking at CoQA results (`explore_dev.py`, CoQA-only by assertion). They are
tested unchanged on TriviaQA and NQ-Open, which are held out.

## What CoQA showed (development set)

All on CoQA, AUROC with paired-bootstrap 95% CIs:
- Lexical Similarity scores 85.3 and EigenScore 80.1 (+5.1 [+2.9, +7.5]).
- The pre-registered MLES is no different from EigenScore (−0.1 [−0.2, +0.0]).
- The rank fusion of EigenScore, Lexical Similarity and LN-Entropy scores 84.6
  (+4.5 [+2.8, +6.3] over EigenScore). It ties Lexical Similarity (−0.7 [−2.1, +0.8])
  rather than beating it.
- The hybrid M3 (MLES + LN-Entropy) scores 82.0 (+1.9 [+0.1, +3.9]).
- Spearman correlations between the three signal families are 0.67 to 0.85, so they are
  partly complementary.

## Frozen hypotheses (held-out test on TriviaQA and NQ-Open, 1,000 questions each)

Notation: ES = EigenScore at index 16. LexSim = Lexical Similarity (mean pairwise
ROUGE-L, negated). LNE = LN-Entropy.

- **H1 · Reversal.** AUROC(LexSim) > AUROC(ES).
- **H2 · Tri-signal fusion** (internal + surface + token):
  TRI = [rank(ES) + rank(LexSim) + rank(LNE)] / 3, with equal weights, label-free and
  untuned.
  - **H2a:** AUROC(TRI) > AUROC(ES).
  - **H2b:** AUROC(TRI) ≥ AUROC(LexSim). This is checked as non-inferiority with a margin
    of 1.0 AUROC point: the lower 95% bound of TRI − LexSim must be > −1.0. We also report
    whether TRI strictly beats LexSim (lower bound > 0).
- **H3 · Hybrid replication.** AUROC(M3 HYB) > AUROC(ES).

## Testing rules

- Each hypothesis is tested separately on each held-out dataset: a paired bootstrap with
  10,000 resamples and seed 2023, giving one-sided p = P(Δ ≤ 0).
- Holm–Bonferroni correction is applied across the 6 superiority tests
  (H1, H2a and H3 on 2 datasets). H2b is reported with its CI.
- **A hypothesis counts as confirmed only if it holds on both TriviaQA and NQ.** Holding on
  one dataset is reported as partial.
- Nothing in this file may be changed after the TriviaQA/NQ analysis is run. Any further
  idea would need a new held-out set.

## Addendum: H4, consistency-then-confidence

Committed after a second CoQA-only exploration (`explore_mode.py`, `explore_lexi.py`) and
still **before any TriviaQA/NQ result was computed**.

**Rationale.** AUROC = P(s₁ > s₀) + ½·P(s₁ = s₀). Lexical Similarity, the best single
signal on CoQA, is exactly tied on 58% of CoQA questions, mostly where all K samples are
identical. Breaking ties with a secondary signal whose AUROC *within* the tied pairs is a
leaves every untied pair unchanged and adds exactly (a − ½) × P(tied pos/neg pair).

**CoQA evidence.**
- Within LexSim ties, LN-Entropy has a = 0.731 while EigenScore has a = 0.590: once the
  answers agree, token confidence still separates right from wrong, but internal spread
  barely does.
- The gain is +0.4 [+0.1, +0.8] over LexSim. That matches the prediction
  (0.731 − ½) × 1.9% ≈ +0.44.

**Method.**
- CTC = dense_rank(LexSim) + ½·(rank(LNE) − ½)/n. This orders questions by LexSim and
  breaks exact ties by LN-Entropy.
- It is fixed and untuned. LN-Entropy was chosen over perplexity, which performed the
  same on CoQA (a = 0.731 vs 0.733), because it uses all K samples rather than only the
  greedy answer.

**H4 (held-out, both datasets, included in the Holm family, which now has 10 tests):**
- **H4a:** AUROC(CTC) > AUROC(LexSim).
- **H4b:** AUROC(CTC) > AUROC(ES).
- **Secondary prediction:** the gain over LexSim is larger on TriviaQA/NQ than on CoQA,
  because lower accuracy means more tied wrong-vs-right pairs. We report P(tied pair) and
  the within-tie a for each dataset.

## Addendum 2: H5, probability-weighted mode mass (PMM)

Committed on 2026-11-13 after a third CoQA-only exploration (`explore_mass.py`), and still
**before any TriviaQA/NQ result was computed**. NQ generation was still running.

**Rationale.** The greedy answer is correct roughly when the meaning it expresses carries
most of the model's probability mass. Counting how many of the K samples agree with the
greedy answer estimates that mass, but only in steps of 1/K. The model also gives the exact
probability p(s) of every distinct string it produced, so the mass can be estimated from
those probabilities directly.

**Method.**
- Take the distinct strings among the greedy answer and the K samples, each with
  p(s) = exp(sum of its token log-probabilities) at temperature 1.
- PMM = Σ p(s) over strings that agree with the greedy answer (ROUGE-L > 0.5, greedy
  included) ÷ Σ p(s) over all distinct strings.
- **PMF = [rank(1 − PMM) + rank(LexSim)] / 2**, equal weights, untuned.

**CoQA evidence (dev).** PMF 87.1 vs LexSim 85.3 (+1.8 [+0.8, +2.8]), vs CTC +1.4
[+0.3, +2.4], vs EigenScore 80.1. PMM alone is 85.8.

**H5 (held-out, both datasets, added to the Holm family, which now has 14 tests):**
- **H5a:** AUROC(PMF) > AUROC(LexSim).
- **H5b:** AUROC(PMF) > AUROC(ES).
