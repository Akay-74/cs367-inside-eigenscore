# Theory note: why each extension should (or should not) work

This note gives the reasoning behind each method in PLAN.md and PLAN_EXPLORATORY.md,
together with the conditions under which each one should fail. References are listed in
LITERATURE.md.

## 1. What EigenScore measures

For one question, the K sampled answers give embeddings z_1..z_K at a layer l. EigenScore
is (1/K) log det(Σ_l + αI), where Σ_l is their K × K covariance. INSIDE's Remark 1 links
this to the differential entropy of a Gaussian, H = ½ log det Σ + const. So EigenScore is
a log-**volume**: it estimates how much semantic space the model's answers occupy.

Two properties follow:

- **(P1) Permutation and rotation invariance.** Relabelling the samples, or rotating the
  embedding space, leaves the eigenvalues unchanged. The score therefore cannot tell
  *how* the answers are arranged, only how much room they take up.
- **(P2) Single-layer dependence.** The value depends on which layer is read. INSIDE's
  own ablation (its Fig. 3b) and LLM-Check both report that detection quality changes
  across layers.

## 2. M1, multi-layer EigenScore (MLES)

**Argument.**
- ES(l) at each layer in the band is a noisy estimate of the same quantity, namely the
  semantic uncertainty of the answer distribution, read at different stages of
  processing.
- Averaging B estimates reduces variance by a factor between 1/B (independent errors)
  and 1 (perfectly correlated errors).
- Adjacent transformer layers are strongly correlated, because the residual stream
  changes incrementally. The variance reduction is therefore real but well short of 1/B.

**Selection bias.**
- Picking the best single layer in hindsight overstates its performance (a winner's
  curse): with 33 layers and noisy AUROCs, the maximum is biased upward.
- MLES fixes the band in advance, so it pays no selection cost. That is why PLAN.md
  compares MLES with the best hindsight layer only as an *upper bound*.

**Prediction.** A modest AUROC gain over ES(16), and above all smaller variation across
datasets and models. If one layer is far better than its neighbours on every dataset,
averaging will hurt.

**Related.** Supervised probe work (Lyapunov probes, LayerMix, CLAP) finds that pooling
over a band of layers matches or beats the best single layer. MLES asks whether the same
holds **without training**.

## 3. M2, normalised embeddings

**Argument.**
- Sun et al. (2024) document "massive activations": a few fixed dimensions in the middle
  layers whose values are orders of magnitude above the rest.
- In a raw covariance, such dimensions can dominate the eigenvalues regardless of
  meaning.
- Normalising each embedding (M2a) or standardising each dimension (M2b) gives every
  direction a comparable say.

**Caveat.**
- Sun et al. find massive activations mostly at the start token and at delimiter tokens.
- Our embeddings come from the **last answer token**, so the effect there may be much
  weaker. In that case M2 should be roughly neutral.
- M2b uses statistics from *previous* questions only, so its first few questions use
  poorly estimated statistics.

## 4. M3, hybrid with LN-Entropy

**Argument.**
- EigenScore (spread of internal states) and LN-Entropy (token-level confidence) are
  computed from different signals.
- If their errors are only partly correlated, an equal-weight rank average can beat both.
  This is the standard ensemble argument: the gain grows as the correlation between the
  two detectors' errors falls.

**Limit.**
- PLAN.md motivates the hybrid with *self-consistent* hallucinations, where the same
  wrong answer is sampled ten times.
- In that case the model is also confident token by token, so LN-Entropy is low too.
  Both detectors then miss the same cases.
- The hybrid is therefore expected to help **on average**, through complementary errors,
  rather than on self-consistent hallucinations specifically. The self-consistent
  analysis in PLAN.md tests this directly.

## 5. X1, cross-depth stability (XDS), the new concept

### Idea
- By (P1), EigenScore discards the *arrangement* of the answers. XDS keeps it.
- At each layer it records the 45 pairwise distances between the 10 answers. It then asks
  how strongly that pattern carries over from one layer to the next: a correlation
  between distance patterns, i.e. representational similarity analysis (RSA;
  Kriegeskorte et al., 2008).

### Why stability should mark knowledge
- **Residual-stream view.** Each layer adds an update to a running representation (Elhage
  et al., 2021). Analyses of factual recall (Geva et al., 2023) describe the subject's
  attributes being retrieved in early-to-middle MLP layers and then read out later.
- **Known fact.** The samples share that retrieved content. Their differences are mostly
  surface variation (wording, length), which the following layers carry along unchanged.
  The arrangement is stable.
- **Confabulation.** There is no shared retrieved content to anchor the samples. Each
  layer's update pushes them in directions that are not coordinated across layers, so
  the arrangement is re-formed as depth increases.

### Independence from EigenScore
- XDS is invariant to how *large* the spread is: multiplying all distances at a layer by a
  constant leaves the correlation unchanged.
- EigenScore is invariant to how the spread is *arranged*.
- Two questions can therefore have equal EigenScore and very different XDS, and vice
  versa. Complementarity is measured directly as Spearman rho(MLES, XDS).

### When XDS should fail
- **Exact duplicates.** When most samples are identical strings, their distances are
  identically zero at every layer, so the pattern looks stable by construction.
  Consistent questions will then get low XDS for a trivial reason, which inflates the
  correlation with consistency-based scores.
- **Too few distinct answers.** With few distinct answers, the correlation is estimated
  from very few distinct values and becomes noisy.
- **No generalisation.** XDS is exploratory. A positive result must be confirmed on fresh
  questions before being claimed (PLAN_EXPLORATORY.md).

## 6. Threats to validity, all methods

- **Labels.** Correctness is ROUGE-L > 0.5 against the reference answers. Janiak et al.
  (2025) show that ROUGE labels have low precision and inflate the AUROC of
  sampling-based detectors, EigenScore included. Absolute AUROCs here are therefore
  optimistic. Paired *differences* between methods are less affected, but not immune.
- **Length.** The same study finds that answer-length statistics alone are competitive
  baselines. Gains should be checked against length.
- **One model, 4-bit.** Llama-2-7B in NF4 only. Quantisation may change the activation
  statistics that M2 relies on.
- **Sample size.** 1,000 questions per dataset. At high model accuracy (CoQA ≈ 80% so
  far), the number of incorrect answers, and with it the precision of AUROC, is lower
  than the sample size suggests.
