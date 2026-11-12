# Related work for the EigenScore extension

Each entry has a **verification level**:
- **[A]** Title, authors and venue were checked online. The listed claim was checked
  against the abstract or landing page.
- **[S]** Found through web search. The claim comes from a search summary or a secondary
  page. Check it against the paper before citing any number.
- **[M]** From memory. Find and check the paper before citing it.

Checked on 2026-11-12. Full PDFs have not yet been read for any entry.

## The method we build on

| Paper | Venue | Relevance | Level |
|---|---|---|---|
| Chen et al., *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection* | ICLR 2024 | EigenScore and feature clipping. Its Fig. 3b shows layer sensitivity. | A |

## Hidden-state / spectral detectors (closest work)

| Paper | Venue | Relevance | Level |
|---|---|---|---|
| Sriramanan et al., *LLM-Check: Investigating Detection of Hallucinations in LLMs* | NeurIPS 2024 | Its hidden-state score is a mean log-determinant of hidden-state covariance, in the single-response setting. Reports that performance oscillates across layers. Closest prior work to MLES. | A (abstract) / S (definition details) |
| Zhang et al., *ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection* | ACL 2025 | Uses layer-to-layer residual updates of a single response, with a trained probe. Closest prior work to XDS, but supervised and single-sample. | S |
| *Hallucination Detection in LLMs Using Spectral Features of Attention Maps* (arXiv 2502.17598) | EMNLP 2025 | Eigenvalues of attention-map Laplacians as probe features. | S |
| *Hallucination Is Linearly Decodable from Mid-Layer Hidden States in Quantized LLMs* (Aiersilan, arXiv 2606.02628) | preprint, May 2026 | Uses 4-bit NF4 7–8B models. Linear probes peak at blocks 13–18 of 32. Supports reading the middle layers of 4-bit models. | A |
| *HIDE and Seek* (arXiv 2506.17748) | preprint 2025 | Uses EigenScore as a training-free baseline. | S |

## Multi-layer aggregation (supervised)

| Paper | Venue | Relevance | Level |
|---|---|---|---|
| *Lyapunov Probes for Hallucination Detection* (arXiv 2603.06081) | preprint 2026 | Aggregating across layers beats the best single layer. | S |
| *The Hallucination Signal Is a Mean Shift* / LayerMix (arXiv 2608.28930) | preprint 2026 | The signal spans a contiguous band of layers, and aggregation matches the oracle layer. | S |
| *Cross-Layer Attention Probing (CLAP)* (arXiv 2509.09700) | preprint 2025 | A trained cross-layer probe over sampled responses. | S |
| *Do Androids Know They're Only Dreaming of Electric Sheep?* (arXiv 2312.17249) | 2023/24 | Single-layer probe salience peaks around layers 10–20. | S |

## Sampling-based uncertainty

| Paper | Venue | Relevance | Level |
|---|---|---|---|
| Farquhar, Kossen, Kuhn, Gal, *Detecting hallucinations in LLMs using semantic entropy* | Nature 630, 2024 | Semantic entropy is the main sampling-based competitor and should be a baseline in a paper. | A |
| Kossen et al., *Semantic Entropy Probes* (arXiv 2406.15927) | preprint 2024 | A cheap probe that approximates semantic entropy from hidden states, on Llama-2-7B. | A |
| Kuhn, Gal, Farquhar, *Semantic Uncertainty* | ICLR 2023 | The original semantic entropy paper. | M |
| Manakul et al., *SelfCheckGPT* | EMNLP 2023 | Sample-consistency detection without internal states. | M |
| Malinin & Gales, *Uncertainty Estimation in Autoregressive Structured Prediction* | ICLR 2021 | Source of LN-Entropy. | M |

## Representations and interpretability (theory for M2 and XDS)

| Paper | Venue | Relevance | Level |
|---|---|---|---|
| Sun, Chen, Kolter, Liu, *Massive Activations in Large Language Models* | COLM 2024 | A few fixed dimensions with huge values in the middle layers, mostly at start and delimiter tokens. Motivates M2. | A |
| Kriegeskorte, Mur, Bandettini, *Representational similarity analysis* | Front. Syst. Neurosci. 2008 | The RSA method used by XDS. | M |
| Kornblith et al., *Similarity of Neural Network Representations Revisited* (CKA) | ICML 2019 | CKA, our first formulation of XDS, later replaced by RSA. | M |
| Geva et al., *Dissecting Recall of Factual Associations in Auto-Regressive LMs* | EMNLP 2023 | Factual recall happens in stages across depth. Supports the XDS hypothesis. | M |
| Elhage et al., *A Mathematical Framework for Transformer Circuits* | Anthropic 2021 | The residual-stream view. | M |

## Evaluation validity

| Paper | Venue | Relevance | Level |
|---|---|---|---|
| Janiak et al., *The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs* (arXiv 2508.08285) | EMNLP 2025 (per search; arXiv says "under review") | ROUGE labels have low precision and inflate detector AUROC, by up to 45.9% under LLM-as-judge. Length heuristics rival complex detectors. **The main threat to our evaluation.** | A (abstract) |
| *OpenHalDet: A Unified Benchmark for Hallucination Detection* (arXiv 2606.06959) | preprint 2026 | Benchmark implementation of EigenScore. It defaults to the last layer. | S |

## Gap

As of this search, no published work does either of the following:
1. Aggregate EigenScore over a band of layers **without training**.
2. Measure how the **arrangement** of the K sampled answers changes across depth, which
   is XDS.

This claim rests on web searches, not a systematic review. Before submitting, check it on
Google Scholar or Semantic Scholar, starting from the papers that cite INSIDE.
