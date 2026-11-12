"""Generation + feature dump for the EigenScore extension study.

For every question we store everything needed to compute *all* detectors
offline from the same samples (so comparisons between them are paired):

  * greedy ("most likely") answer -> correctness label (ROUGE-L > 0.5, the
    paper's AUC_r rule) and its perplexity
  * K sampled answers (paper settings: T=0.5, top-p 0.99, top-k 5)
  * for every sample: last-answer-token hidden state at EVERY layer, plus the
    sum of token log-probs and the answer length (for LN-Entropy)

The hidden states are re-computed with a clean forward pass over the answer
on top of the prompt's KV cache, so "last token" is exact and padding cannot
leak in. The prompt is run once per question and the cache is cropped back
after every answer, which keeps a 1k-token CoQA story inside 8 GB.

Outputs (for --out results/llama2_nq):
  results/llama2_nq.emb.npy   (n, K, L+1, d) float16 memmap, written per question
  results/llama2_nq.npz       per-question scalars (log-probs, lengths, rouge, ...)
  results/llama2_nq.json      args, questions, references, greedy + sampled texts

    python run_extension.py --dataset nq --n 1000 --out results/llama2_nq
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

FEWSHOT = (
    "Answer the question with a short phrase.\n\n"
    "Q: What is the capital of France?\nA: Paris\n\n"
    "Q: Who wrote the play Romeo and Juliet?\nA: William Shakespeare\n\n"
    "Q: How many legs does a spider have?\nA: eight\n\n"
    "Q: In which year did the Second World War end?\nA: 1945\n\n"
    "Q: What is the chemical symbol for gold?\nA: Au\n\n"
)
COQA_JSON = Path(__file__).resolve().parents[1] / "data/datasets/coqa-dev-v1.0.json"
KV_BUDGET = 1.5e9   # bytes of KV cache allowed per generate() call


def load_questions(name, n, seed, coqa_json=COQA_JSON):
    """List of (prompt, question, references)."""
    if name == "coqa":
        # Official dev file: one item per turn, story + the previous gold Q/A turns.
        data = json.load(open(coqa_json))["data"]
        rows = []
        for s in data:
            hist = ""
            for i, (q, a) in enumerate(zip(s["questions"], s["answers"])):
                refs = [a["input_text"]] + [s["additional_answers"][k][i]["input_text"]
                                            for k in sorted(s.get("additional_answers", {}))]
                prompt = s["story"] + "\n\n" + hist + f"Q: {q['input_text']}\nA:"
                rows.append((prompt, q["input_text"], list(dict.fromkeys(refs))))
                hist += f"Q: {q['input_text']}\nA: {a['input_text']}\n\n"
    else:
        from datasets import load_dataset
        if name == "nq":
            ds = load_dataset("google-research-datasets/nq_open", split="validation")
            qa = [(r["question"], list(r["answer"])) for r in ds]
        elif name == "triviaqa":
            ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
            qa = [(r["question"], list(dict.fromkeys([r["answer"]["value"]] + r["answer"]["aliases"])))
                  for r in ds]
        else:
            raise ValueError(name)
        rows = [(FEWSHOT + f"Q: {q}\nA:", q, refs) for q, refs in qa]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))[:n]
    return [rows[i] for i in idx]


def cut_answer(tok, ids):
    """Keep generated ids up to the first newline / EOS."""
    out = []
    for t in ids:
        if t in (tok.eos_token_id, tok.pad_token_id):
            break
        if "\n" in tok.decode([t]):
            break
        out.append(int(t))
    return out


@torch.no_grad()
def features(model, prompt_ids, answers):
    """Per answer: hidden state of its last token at every layer, sum log-prob, length.

    The prompt is encoded once; each answer is run on top of its KV cache, which
    is then cropped back to the prompt. Equivalent to a full forward over
    prompt + answer (checked by --check_features).
    """
    dev = model.device
    P = len(prompt_ids)
    out = model(input_ids=torch.tensor([prompt_ids], device=dev), use_cache=True,
                output_hidden_states=True, logits_to_keep=1)
    cache = out.past_key_values
    first_logp = torch.log_softmax(out.logits[0, -1].float(), -1)
    prompt_last = torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()
    del out

    n_layers, d = prompt_last.shape
    embs = np.zeros((len(answers), n_layers, d), dtype=np.float16)
    lp_sum, lens = np.zeros(len(answers)), np.zeros(len(answers), dtype=int)
    for i, a in enumerate(answers):
        if not a:                       # empty answer: last token is the prompt's
            embs[i] = prompt_last
            continue
        o = model(input_ids=torch.tensor([a], device=dev), past_key_values=cache,
                  use_cache=True, output_hidden_states=True)
        embs[i] = torch.stack([h[0, -1] for h in o.hidden_states]).float().cpu().numpy()
        lp = first_logp[a[0]]
        if len(a) > 1:
            rest = torch.log_softmax(o.logits[0, :-1].float(), -1)
            lp = lp + rest.gather(1, torch.tensor(a[1:], device=dev)[:, None]).sum()
        lp_sum[i], lens[i] = float(lp), len(a)
        del o
        cache.crop(P)
    return embs, lp_sum, lens


@torch.no_grad()
def features_full(model, prompt_ids, answers):
    """Reference implementation: one plain forward per prompt + answer."""
    rows = []
    for a in answers:
        s = prompt_ids + a
        o = model(input_ids=torch.tensor([s], device=model.device), output_hidden_states=True)
        e = torch.stack([h[0, -1] for h in o.hidden_states]).float().cpu().numpy()
        lp = torch.log_softmax(o.logits[0].float(), -1)[len(prompt_ids) - 1:len(s) - 1]
        lp = lp.gather(1, torch.tensor(a, device=model.device)[:, None]).sum() if a else 0.0
        rows.append((e, float(lp)))
    return rows


def load_model(name, load_in_4bit):
    tok = AutoTokenizer.from_pretrained(name)
    kw = dict(dtype=torch.float16, device_map="cuda:0")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(   # same config as the midsem run
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    return tok, model


def gen_batch_size(model, P, k):
    c = model.config
    kv_per_tok = 2 * c.num_hidden_layers * c.num_key_value_heads * (c.hidden_size // c.num_attention_heads) * 2
    return int(max(1, min(k, KV_BUDGET // (kv_per_tok * (P + 64)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dataset", default="nq", choices=["nq", "triviaqa", "coqa"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--max_new_tokens", type=int, default=None, help="default 32 for CoQA, 20 otherwise")
    ap.add_argument("--coqa_json", default=str(COQA_JSON), help="official coqa-dev-v1.0.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check_features", action="store_true",
                    help="also run the plain full forward and report the max difference")
    ap.add_argument("--out", required=True, help="output prefix, e.g. results/llama2_nq")
    args = ap.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = 32 if args.dataset == "coqa" else 20
    assert torch.cuda.is_available(), "this script expects a CUDA GPU"

    tok, model = load_model(args.model, args.load_in_4bit)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    qs = load_questions(args.dataset, args.n, args.seed, args.coqa_json)
    n, K = len(qs), args.k
    n_hid, d = model.config.num_hidden_layers + 1, model.config.hidden_size

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    emb_path = out.with_suffix(".emb.npy")
    rec = dict(question=[], refs=[], greedy=[], samples=[], rouge=[], correct=[],
               greedy_lp=[], greedy_len=[], samp_lp=[], samp_len=[], prompt_len=[])
    if args.resume and out.with_suffix(".json").exists():
        meta = json.loads(out.with_suffix(".json").read_text())
        sc = np.load(out.with_suffix(".npz"))
        done = len(meta["question"])
        for key in ("question", "refs", "greedy", "samples"):
            rec[key] = meta[key]
        for key in ("rouge", "correct", "greedy_lp", "greedy_len", "samp_lp", "samp_len", "prompt_len"):
            rec[key] = list(sc[key][:done])
        emb = np.lib.format.open_memmap(emb_path, mode="r+")
        assert emb.shape == (n, K, n_hid, d), emb.shape
        print(f"resuming at question {done}/{n}", flush=True)
    else:
        done = 0
        emb = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(n, K, n_hid, d))

    t0, max_diff = time.time(), 0.0
    for qi in range(done, n):
        prompt, q, refs = qs[qi]
        torch.manual_seed(args.seed * 100_003 + qi)        # per question: resumable + reproducible
        p_ids = tok(prompt)["input_ids"]
        x = torch.tensor([p_ids], device=model.device)
        gkw = dict(attention_mask=torch.ones_like(x), max_new_tokens=args.max_new_tokens,
                   pad_token_id=tok.pad_token_id, stop_strings=["\n"], tokenizer=tok)
        g = model.generate(x, do_sample=False, temperature=None, top_p=None, top_k=None, **gkw)
        g_ids = cut_answer(tok, g[0, len(p_ids):].tolist())
        s_ids, b = [], gen_batch_size(model, len(p_ids), K)
        while len(s_ids) < K:
            m = min(b, K - len(s_ids))
            s = model.generate(x, do_sample=True, temperature=0.5, top_p=0.99, top_k=5,
                               num_return_sequences=m, **gkw)
            s_ids += [cut_answer(tok, row[len(p_ids):].tolist()) for row in s]
            del s

        e, lp, ln = features(model, p_ids, s_ids)
        _, glp, gln = features(model, p_ids, [g_ids])
        if args.check_features:
            ref = features_full(model, p_ids, s_ids[:3])
            for j, (re_, rlp) in enumerate(ref):
                rel = np.abs(e[j].astype(np.float32) - re_).max() / (np.abs(re_).max() + 1e-6)
                max_diff = max(max_diff, rel)
                print(f"   check q{qi} s{j}: emb rel diff {rel:.2e}  logp {lp[j]:.3f} vs {rlp:.3f}", flush=True)
        emb[qi] = e
        greedy = tok.decode(g_ids).strip()
        r = max(scorer.score(ref, greedy)["rougeL"].fmeasure for ref in refs) if greedy else 0.0

        rec["question"].append(q); rec["refs"].append(refs); rec["greedy"].append(greedy)
        rec["samples"].append([tok.decode(a).strip() for a in s_ids])
        rec["rouge"].append(r); rec["correct"].append(r > 0.5)
        rec["greedy_lp"].append(glp[0]); rec["greedy_len"].append(gln[0])
        rec["samp_lp"].append(lp); rec["samp_len"].append(ln); rec["prompt_len"].append(len(p_ids))
        if qi % 10 == 0 or qi == n - 1:
            el = time.time() - t0
            acc = np.mean(rec["correct"])
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"[{qi+1}/{n}] {el/(qi+1-done):.1f}s/q  acc={acc:.2f}  P={len(p_ids)} b={b} "
                  f"peak={mem:.1f}GiB  q={q[:50]!r} greedy={greedy!r}", flush=True)
            emb.flush(); save(args, rec, out)
    emb.flush(); save(args, rec, out)
    if args.check_features:
        print(f"max relative embedding difference vs full forward: {max_diff:.2e}")


def save(args, rec, out):
    np.savez(out.with_suffix(".npz"), samp_lp=np.stack(rec["samp_lp"]), samp_len=np.stack(rec["samp_len"]),
             greedy_lp=np.array(rec["greedy_lp"]), greedy_len=np.array(rec["greedy_len"]),
             rouge=np.array(rec["rouge"]), correct=np.array(rec["correct"]),
             prompt_len=np.array(rec["prompt_len"]))
    out.with_suffix(".json").write_text(json.dumps(dict(args=vars(args), question=rec["question"], refs=rec["refs"],
                                                        greedy=rec["greedy"], samples=rec["samples"]), indent=1))


if __name__ == "__main__":
    main()
