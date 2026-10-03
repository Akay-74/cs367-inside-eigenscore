# Changes to the upstream repository

Upstream is [alibaba/eigenscore](https://github.com/alibaba/eigenscore) at `ea8062a`.
Every original file is kept alongside as `*.orig`, and the original README as
`README.upstream.md`.

## Blocking bugs

| # | File | Problem | Fix |
|---|---|---|---|
| 1 | `pipeline/generate.py` | `import dataeval.TruthfulQA` — the module does not exist in the repo, so the entry point raised `ModuleNotFoundError` before doing anything. | Import removed. The `TruthfulQA` branches further down are inert but harmless. |
| 2 | `func/metric.py` | `np.float` used in 14 places; removed in NumPy ≥ 1.24. Every `getEigenIndicator*` variant crashed. | `np.float` → `np.float64`. |
| 3 | `_settings.py` | `_BASE_DIR` hardcoded to `/disk1/chenchao/Code/hallucination_detection/data`. | Repo-relative defaults, overridable via `EIGENSCORE_DATA_DIR` / `EIGENSCORE_MODEL_PATH` / `EIGENSCORE_OUTPUT_DIR`. |
| 4 | `pipeline/generate.py` | Log file opened at `./data/output/...` before that directory is created. | Directory created first; path now derives from `_settings.GENERATION_FOLDER`. |
| 5 | `func/evalFunc.py` | `from metric import *` / `from plot import *` — only importable with cwd inside `func/`. | Changed to `from func.metric import *` / `from func.plot import *`. |
| 6 | `func/metric.py` | Imported `selfcheckgpt` at module load, but only used it in `get_sent_scores_bertscore()`, which nothing calls. A heavy dependency for dead code. | Import moved inside the function. |
| 7 | `pipeline/generate.py` | Built a `torchmetrics` `BERTScore` on `device="cuda"` from a local `./data/weights/bert-base/` path. Its consumer `getAvgBertScore()` adds `0` in a loop — upstream disabled it as "too slow". | Construction removed; the disabled metric still reports its constant. |
| 8 | `dataeval/{triviaqa,nq_open,load_worker}.py` | `import ipdb` at module scope — a debugger, pulled in on the critical path. | Commented out. |
| 9 | `func/plot.py` | `plt.savefig("./Figure/...")` into a directory that doesn't exist, then `plt.show()`, which blocks without a display. | `Agg` backend, `makedirs`, no `show()`. Output dir via `EIGENSCORE_FIGURE_DIR`. |
| 10 | `func/evalFunc.py` | `file_name.split("_")[1]` — `IndexError` on any path without an underscore. | Passes the whole path; `VisAUROC` already pattern-matches the dataset name. |
| 11 | `func/metric.py` | `.to("cuda")` hardcoded in 16 places, ignoring `--device`. | Derives the device from the hidden states via `_device_of()`. |
| 12 | `dataeval/load.py` | `DEFAULT_DEVICE = 'cuda:7'`. | Left as-is — this module is off the generation path and needs `persist_to_disk`. |

## Missing from the paper

**Test time feature clipping (§3.2) was not implemented anywhere in the repo.** The only
`*Clip*` functions present are `ParameterClip{,_v1,_v2}`, which mask `lm_head` weights
using `.npy` files that aren't shipped — a different, unused experiment.

Added `func/feature_clip.py`:

* `FeatureClipper` — a rolling memory bank of `N` token activations (default 3000, the
  paper's value) with per-neuron thresholds at the bottom/top `p`-th percentile
  (default 0.2%, the paper's value), applying Eq. 8 elementwise.
* Applied via a forward hook on the **penultimate decoder layer**, so the clip is active
  during autoregressive decoding and actually changes what the model generates. This
  matters: the point of §3.2 is to suppress overconfident generation, which only happens
  if the clip feeds back into the sampling loop.
* Warm-up guard — pass-through until the bank holds `min_bank` rows, so thresholds are
  never estimated from a handful of tokens.
* `_get_decoder_layers()` handles LLaMA / OPT / Falcon / GPT-2 layouts.

Verified against `numpy.percentile` to ~1e-6, plus ring-buffer wraparound, in-band
invariance, and warm-up behaviour.

## New capabilities

* **Quantization** — `--load_in_4bit` / `--load_in_8bit` (bitsandbytes NF4 / int8), which
  is what lets a 7B model and its hidden states share an 8 GB GPU.
* **Hub model ids** — `models/_load_model.py` resolves a local dir, then a hub id, then a
  short alias. Upstream only accepted local directories under `MODEL_PATH`, and its
  if/elif chain silently left `model` undefined for unrecognized names.
* **`--num_return_sequences`** — generations per forward pass, decoupled from `K`.
  `output_hidden_states=True` retains every layer for every token, which dominates VRAM;
  this lets you keep the paper's `K=10` on a small card.
* **`--max_new_tokens`** — was a hardcoded 256.
* **`--low_memory`** — `output_hidden_states=True` makes `generate()` return every
  layer for every token, prompt included: a 1000-token CoQA story at 33 layers x 5
  sequences is several GB, and it OOM'd an 8GB card partway through CoQA. Since
  EigenScore only ever reads one middle layer's last token, this flag captures that
  layer with a forward hook instead. Verified bit-identical to the full path
  (`tests/test_setup.py::test_low_memory_capture_equivalence`).
* **Eval CLI** — `python -m func.evalFunc <pkl> --correctness {rouge,similarity,exact_match}`.
  Upstream required editing a block of commented-out hardcoded paths in `__main__`.
* **`--feature_clip`, `--clip_percentile`, `--memory_bank_size`** for §3.2.
* `requirements.txt`, which upstream did not have.

## Not changed

The EigenScore computation itself (`getEigenIndicator_v0`) is untouched: middle layer
`int(L/2)`, last-token embedding, `α=1e-3`, `mean(log10(svd))`. Note it uses `log10`
where Eq. 6 writes `log` — a constant factor of `ln(10)` that cannot change AUROC or the
sign of PCC, so it is left alone to match published numbers.
