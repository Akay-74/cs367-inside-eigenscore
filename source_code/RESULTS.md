# Reproduction results

Setup differs from the paper in three ways that matter, all forced by an 8 GB GPU:
4-bit NF4 weights instead of fp16, Llama-2-7B instead of LLaMA-1-7B, and 250 CoQA
questions instead of the full 7983-pair split. So the claim being tested is whether the
**ordering** of methods reproduces, not whether the exact numbers do.

```
python -m pipeline.generate --model llama2-7b-hf --dataset coqa \
    --load_in_4bit --low_memory --feature_clip --no_sentsim \
    --num_generations_per_prompt 10 --num_return_sequences 2 \
    --fraction_of_data_to_use 0.03132 --seed 2023 --project_ind 10
python -m func.evalFunc data/output/llama2-7b-hf_coqa_10/0.pkl --correctness rouge
```

K=10, temperature 0.5, top-p 0.99, α=1e-3, middle-layer last-token embedding, p=0.2%,
N=3000 — all as specified in §4.1. Correctness is ROUGE-L > 0.5, the paper's AUC_r.
Model accuracy on this subset: 36.8%.

## AUROC, CoQA, 250 questions

| Method | Paper (LLaMA-7B, AUC_r) | This run (Llama-2-7B 4-bit) | +FC |
|---|---|---|---|
| Energy | 54.7 | **54.7** | 53.1 |
| Perplexity | 68.3 | 72.3 | 71.4 |
| LN-Entropy | 73.6 | 77.5 | 77.3 |
| Lexical Similarity | 77.8 | 80.5 | 79.3 |
| **EigenScore** | **80.8** | **83.5** | 82.5 |

**Read the EigenScore row carefully.** The paper's Table 1 EigenScore already includes feature
clipping: its Table 2 lists EigenScore without clipping at 79.3 AUC_s and with clipping at 80.4
AUC_s, and 80.4 is the AUC_s printed in Table 1. The Table 1 baselines are without clipping. So
the like-for-like number for the paper's 80.8 is our **+FC run, 82.5**; the paper does not report
an AUC_r for EigenScore without clipping. Also note the paper samples with top-k 5; this run used
the repository default of 10.

**The ranking reproduces exactly**, in the paper's order, with EigenScore best and Energy
— an OOD-detection method not designed for this — barely above chance. Absolute values
land within a few points despite the quantization and model change.

## Feature clipping did not reproduce

The paper's Table 2 reports feature clipping improving EigenScore on CoQA by +1.1 AUROC
(79.3 → 80.4), measured as AUC_s (sentence-similarity correctness). We could only measure AUC_r
(ROUGE-L correctness), because the nli-roberta-large model was not available, so the two numbers
are not strictly the same measurement. We measured the opposite sign, and the effect is not resolvable at this
sample size:

```
EigenScore, feature clipping effect (paired bootstrap over questions, 10k resamples,
both runs on the identical 250 questions with the same seed):
  delta AUROC = -0.0102
  95% CI      = [-0.0468, +0.0263]
  P(FC helps) = 0.30
```

The confidence interval comfortably contains both zero and the paper's claimed +0.011, so
this run neither confirms nor contradicts §3.2 — it lacks the resolution to do either. A
1-point effect needs far more than 250 questions to detect; the paper used 7983. The
clipping itself is verifiably active (13,749,839 activations clipped over 1,159,186 tokens × 4,096 neurons = 0.29%;
at most ~0.4% would be expected from p=0.2% per tail) and its thresholds are unit-tested against
`numpy.percentile`, so this is a statistical power limitation, not evidence the
implementation is wrong.

## EigenScore vs. the strongest baseline

Lexical Similarity is the paper's best-performing baseline, so it is the comparison that
carries the claim:

```
EigenScore - Lexical Similarity = +0.0293 AUROC
  95% CI [-0.0057, +0.0651], P(>0) = 0.95
```

Directionally consistent with the paper's +3.0 on CoQA, and just short of excluding zero
at 250 questions.

## Reproducing at full scale

`--fraction_of_data_to_use 1` runs all 7983 pairs. At the ~6 s/question observed here
that is roughly 13 hours per configuration on this GPU.
