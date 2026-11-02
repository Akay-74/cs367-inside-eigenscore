# EigenScore extension: study plan

Goal: improve INSIDE's EigenScore for LLM hallucination detection without any
training, and show the gain with paired statistics on the same samples.
Everything below is fixed **before** looking at results.

## 1. Weaknesses being addressed

| Weakness of EigenScore | Fix |
|---|---|
| Reads one hand-picked layer (`int(L/2)`). The paper's Fig. 3b shows AUROC changes by several points across layers. | Multi-layer EigenScore |
| Raw covariance is dominated by a few very large ("massive") activation dimensions, not by meaning. | Normalised EigenScore |
| Blind to self-consistent hallucinations: the same wrong answer 10 times gives a low score. Feature clipping gave only +1.1 in the paper and was inconclusive in our midsem run. | Confidence-aware hybrid |

## 2. Methods (all computed offline from the same samples)

Notation: for each question there are K = 10 sampled answers. `e[k, l]` is
answer k's hidden state at the **last answer token**, at hidden-state index l
(index 0 = embeddings, 1..L = decoder layers; Llama-2-7B: L = 32, 33 entries).
`ES(l)` = original EigenScore at index l:
`mean(log10(svd(cov(E_l) + 1e-3·I)))`, where `E_l` is the K × d matrix of
`e[:, l]` and `cov` is over the d features, as in `func/metric.py:getEigenIndicator_v0`.

**Baselines**
- Perplexity: of the greedy answer.
- LN-Entropy: mean over the K samples of (−sum log-prob / length).
- Lexical Similarity: mean pairwise ROUGE-L over the K samples (higher = more consistent, so negate it).
- **EigenScore (original):** `ES(L/2)`, i.e. index 16 for Llama-2-7B.

**M1. Multi-layer EigenScore (primary method)**
- `MLES = mean_{l in band} ES(l)`
- **Band fixed in advance:** `l = floor(L/4) .. floor(3L/4)`, inclusive, i.e. the middle half of the network. For Llama-2-7B that is indices **8..24** (17 layers).
- No per-layer weighting and no tuning.

**M2. Normalised EigenScore**
- **M2a (primary variant):** L2-normalise each `e[k, l]` to unit length before the covariance. Applied both single-layer (index L/2) and multi-layer (band above).
- **M2b (secondary):** standardise each dimension with mean/std estimated from the embeddings of *previous* questions only (a running bank, like the feature-clipping memory bank; no leakage from the current question), then compute ES.

**M3. Confidence-aware hybrid**
- `HYB = (rank(MLES) + rank(LN-Entropy)) / 2`
- Ranks are taken within each dataset, both oriented so that higher = more likely hallucinated.
- Equal weights; label-free, no training.
- Secondary variant: `MLES + perplexity`.

**Stretch: adaptive K.** Sample one answer at a time and stop once ES is clearly low or high. Report AUROC against the average number of samples. Only if time allows.

## 3. Experimental setup

| Item | Setting |
|---|---|
| Model | Llama-2-7B (`NousResearch/Llama-2-7b-hf`), 4-bit NF4 (bitsandbytes), RTX 4060 8 GB |
| Datasets | CoQA dev (story + previous turns + question), TriviaQA `rc.nocontext` validation, NQ-Open validation |
| Sample size | **1,000 questions per dataset** (random subset, seed 2023) |
| Sampling | K = 10, temperature 0.5, top-p 0.99, top-k 5 (paper value; midsem used 10) |
| Max new tokens | 20 for TriviaQA/NQ, 32 for CoQA; stop at newline |
| Correctness | greedy answer, ROUGE-L F > 0.5 against any reference (paper's AUC_r) |
| Features saved | last-answer-token hidden state at **every** layer for every sample, sum log-prob, length, texts |

## 4. Evaluation

- AUROC for every method, per dataset (label 1 = incorrect).
- **Primary comparison (one, pre-registered):** MLES vs original EigenScore, using a paired bootstrap over questions (10,000 resamples). We claim an improvement only if the 95% CI of the AUROC difference is above 0.
- **Secondary comparisons:**
  - M2a, M2b, HYB vs original EigenScore;
  - HYB vs MLES.
  - Reported with CIs, and labelled as secondary.
- **Analyses:**
  - per-layer AUROC curve, `ES(l)` for every l: shows single-layer fragility and where the band sits;
  - self-consistent hallucinations: the incorrect answers whose samples agree (lexical similarity ≥ 0.9). Compare the fraction each method flags at a fixed 20% false-positive rate;
  - robustness: the band's AUROC vs the best single layer chosen in hindsight (an upper bound, not a fair method).

## 5. Expected outcomes (hypotheses, not results)

- **MLES:** +1 to +3 AUROC over EigenScore, and smaller variance across datasets.
- **M2:** uncertain; may be neutral.
- **HYB:** largest gain on TriviaQA/NQ, where the paper shows perplexity competitive with EigenScore.
- **Negative results:** reported with their CIs.

## 6. Status / TODO for the GPU run

- [x] `run_extension.py`: generation + all-layer feature dump. Tested on CPU only (Qwen2.5-1.5B, 5 NQ questions, about 22 s/q). Defaults are still CPU/Qwen.
- [ ] Add `--device cuda`, `--load_in_4bit`, model default `NousResearch/Llama-2-7b-hf`; remove `torch.set_num_threads(4)` on GPU.
- [ ] Add a CoQA loader (HF `stanfordnlp/coqa` validation; build one item per turn with story + previous Q/A turns).
- [ ] Smoke test: 5 questions per dataset; check accuracy is not near 0 or 1 and that last-token indices are correct.
- [ ] Full runs: 1,000 per dataset. Rough time on a 4060: ~1.5–2 h for CoQA, less for TriviaQA/NQ.
- [ ] `analyze_extension.py`: all methods above, AUROC, bootstrap, per-layer plot, self-consistent subset.
- [ ] Write results to `RESULTS_EXTENSION.md`.

`run_extension.py` imports nothing from the repo, only `torch`, `transformers`, `datasets`, `rouge_score` and `numpy`.
