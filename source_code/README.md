# INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection (ICLR 2024)

Working setup of the [official EigenScore repository](https://github.com/alibaba/eigenscore)
for the paper [INSIDE](https://arxiv.org/pdf/2402.03744) (Chen et al., ICLR 2024).

The upstream repo is a code drop: its README's Setup / Requirements / Run sections are
empty headers, paths point at the authors' machine, and the paper's second contribution
(test time feature clipping, §3.2) is not implemented. This fork fixes that — see
[CHANGES.md](CHANGES.md) for the full list of repairs.

## What the method does

Two ideas, both operating on the LLM's **internal states** rather than its logits or its
decoded text:

1. **EigenScore** (§3.1) — sample `K` responses to one question, take each response's
   sentence embedding from the middle layer, and score semantic divergence as the
   log-determinant of their covariance matrix:

   ```
   E(Y|x,θ) = (1/K) · log det(Σ + α·I) = (1/K) · Σᵢ log λᵢ
   ```

   Confident answers are semantically tight, so the eigenvalues collapse toward 0 and the
   score is low. Hallucinations spread out, so the eigenvalues — and the score — grow.
   This equals the differential entropy in the embedding space (Remark 1).

2. **Test time feature clipping** (§3.2) — clamp each neuron in the penultimate layer to
   the bottom/top `p`-th percentile of a rolling memory bank of `N` token activations
   (Eq. 8). Truncating these extreme activations suppresses overconfident generation,
   which is what catches the *self-consistent* hallucinations that pure consistency
   metrics miss.

## Setup

Requires a CUDA GPU and Python 3.11.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

### Data

CoQA (`coqa-dev-v1.0.json`) and SQuAD (`dev-v2.0.json`) already ship in `data/datasets/`.
TriviaQA and NQ-Open are pulled from the HuggingFace hub on first use.

### Models

Model names resolve in this order: a local directory under `$EIGENSCORE_MODEL_PATH`, then
a HuggingFace hub id (anything containing `/`), then a short alias:

| alias | resolves to |
|---|---|
| `llama-7b-hf` | `huggyllama/llama-7b` |
| `llama-13b-hf` | `huggyllama/llama-13b` |
| `llama2-7b-hf` | `NousResearch/Llama-2-7b-hf` (ungated mirror) |
| `opt-6.7b` | `facebook/opt-6.7b` |
| `falcon-7b` | `tiiuae/falcon-7b` |

You can also pass any hub id directly, e.g. `--model meta-llama/Llama-2-7b-hf`.

### Paths

All paths default to repo-relative and are overridable:

```bash
export EIGENSCORE_DATA_DIR=...     # default <repo>/data
export EIGENSCORE_MODEL_PATH=...   # default <repo>/data/weights
export EIGENSCORE_OUTPUT_DIR=...   # default <repo>/data/output
```

## Run

Generation — samples `K` responses per question and records EigenScore alongside every
baseline (perplexity, energy, length-normalized entropy, lexical similarity):

```bash
python -m pipeline.generate \
    --model llama2-7b-hf --dataset coqa \
    --load_in_4bit \
    --low_memory \
    --feature_clip \
    --fraction_of_data_to_use 0.01 \
    --num_generations_per_prompt 10 \
    --num_return_sequences 5 \
    --project_ind 0
```

Evaluation — AUROC and PCC for every method:

```bash
python -m func.evalFunc data/output/llama2-7b-hf_coqa_0/0.pkl --correctness rouge
```

`--correctness` picks the paper's correctness measure: `rouge` (ROUGE-L > 0.5, the AUC_r
column in Table 1), `similarity` (nli-roberta cosine > 0.9, AUC_s), or `exact_match`.

### Fitting on a small GPU

The paper used full-precision 7B/13B models. On an 8 GB card:

* `--load_in_4bit` — NF4 quantized weights (~4 GB for 7B). `--load_in_8bit` also works.
* `--num_return_sequences` — generations per forward pass. `output_hidden_states=True`
  keeps every layer's activations for every token, which dominates memory; lowering this
  trades speed for VRAM while `--num_generations_per_prompt` stays at the paper's `K=10`.
* `--low_memory` — capture only the middle layer through a hook instead of returning
  every layer for every token. Mathematically identical for EigenScore (there is a test
  asserting this) and the difference between running and running out of memory on a long
  CoQA story.
* `--max_new_tokens` — default 256, as in the paper.
* `--no_sentsim` — skip the nli-roberta model; disables only the EigenScore-Output
  baseline.

### Key flags

| flag | default | paper |
|---|---|---|
| `--num_generations_per_prompt` | 10 | `K = 10` |
| `--temperature` | 0.5 | 0.5 |
| `--top_p` / `--top_k` | 0.99 / 10 | 0.99 / 5 |
| `--clip_percentile` | 0.2 | `p = 0.2` |
| `--memory_bank_size` | 3000 | `N = 3000` |

The regularizer `α = 1e-3` and the middle-layer embedding (`int(L/2)`, last token) are set
in `func/metric.py:getEigenIndicator_v0`, matching §4.1.

## Using EigenScore in your own code

```python
from func.metric import getEigenIndicator_v0

out = model.generate(input_ids, num_return_sequences=10, do_sample=True,
                     output_hidden_states=True, return_dict_in_generate=True)
score, eigenvalues = getEigenIndicator_v0(out.hidden_states, num_tokens)
```

A higher score means more semantic divergence, i.e. more likely hallucinated. Note that
`evalFunc` negates it before computing AUROC, so that larger = more trustworthy.

## Results

A reproduction on 250 CoQA questions is in [RESULTS.md](RESULTS.md): the paper's method
ordering reproduces exactly (EigenScore 83.5 > Lexical Similarity 80.5 > LN-Entropy 77.5 >
Perplexity 72.3 > Energy 54.7 AUROC), while the feature-clipping effect is too small to
resolve at that sample size.

## Citation

```
@article{chen2024inside,
title={INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection},
author={Chen, Chao and Liu, Kai and Chen, Ze and Gu, Yi and Wu, Yue and Tao, Mingyuan and Fu, Zhihang and Ye, Jieping},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024}
}
```
