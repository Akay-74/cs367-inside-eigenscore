"""Generation + feature dump for the EigenScore extension study.

For every question we store everything needed to compute *all* detectors
offline from the same samples (so comparisons between them are paired):

  * greedy ("most likely") answer -> correctness label (ROUGE-L > 0.5, the
    paper's AUC_r rule) and its perplexity
  * K sampled answers (paper settings: T=0.5, top-p 0.99, top-k 5)
  * for every sample: last-answer-token hidden state at EVERY layer, plus the
    sum of token log-probs and the answer length (for LN-Entropy)

The hidden states are re-computed with one clean forward pass over
prompt + answer, so "last token" is exact and padding cannot leak in.

    python run_extension.py --dataset nq --n 300 --out results/qwen15_nq.npz
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


def load_questions(name, n, seed):
    from datasets import load_dataset
    if name == "nq":
        ds = load_dataset("google-research-datasets/nq_open", split="validation")
        rows = [(r["question"], list(r["answer"])) for r in ds]
    elif name == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        rows = [(r["question"], list(dict.fromkeys([r["answer"]["value"]] + r["answer"]["aliases"]))) for r in ds]
    else:
        raise ValueError(name)
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
    """Per answer: hidden state of its last token at every layer, sum log-prob, length."""
    L = model.config.num_hidden_layers + 1
    seqs = [prompt_ids + a for a in answers]
    maxlen = max(len(s) for s in seqs)
    pad = model.config.pad_token_id or 0
    inp = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    att = torch.zeros_like(inp)
    for i, s in enumerate(seqs):
        inp[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    out = model(input_ids=inp, attention_mask=att, output_hidden_states=True)
    logp = torch.log_softmax(out.logits.float(), -1)
    P = len(prompt_ids)
    embs = np.zeros((len(seqs), L, out.hidden_states[0].shape[-1]), dtype=np.float16)
    lp_sum, lens = np.zeros(len(seqs)), np.zeros(len(seqs), dtype=int)
    for i, (s, a) in enumerate(zip(seqs, answers)):
        last = len(s) - 1
        embs[i] = torch.stack([h[i, last] for h in out.hidden_states]).float().numpy()
        if a:
            tgt = torch.tensor(a)
            lp = logp[i, P - 1:P - 1 + len(a)].gather(1, tgt[:, None]).squeeze(1)
            lp_sum[i], lens[i] = float(lp.sum()), len(a)
    return embs, lp_sum, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--dataset", default="nq", choices=["nq", "triviaqa"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    qs = load_questions(args.dataset, args.n, args.seed)
    rec = dict(question=[], refs=[], greedy=[], samples=[], rouge=[], correct=[],
               greedy_lp=[], greedy_len=[], samp_lp=[], samp_len=[], emb=[])
    t0 = time.time()
    for qi, (q, refs) in enumerate(qs):
        prompt = FEWSHOT + f"Q: {q}\nA:"
        p_ids = tok(prompt)["input_ids"]
        x = torch.tensor([p_ids])
        g = model.generate(x, attention_mask=torch.ones_like(x), do_sample=False,
                           max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
                           stop_strings=["\n"], tokenizer=tok)
        g_ids = cut_answer(tok, g[0, len(p_ids):].tolist())
        s = model.generate(x, attention_mask=torch.ones_like(x), do_sample=True, temperature=0.5,
                           top_p=0.99, top_k=5, num_return_sequences=args.k,
                           max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
                           stop_strings=["\n"], tokenizer=tok)
        s_ids = [cut_answer(tok, row[len(p_ids):].tolist()) for row in s]

        emb, lp, ln = features(model, p_ids, s_ids)
        _, glp, gln = features(model, p_ids, [g_ids])
        greedy = tok.decode(g_ids).strip()
        r = max(scorer.score(ref, greedy)["rougeL"].fmeasure for ref in refs) if greedy else 0.0

        rec["question"].append(q); rec["refs"].append(refs); rec["greedy"].append(greedy)
        rec["samples"].append([tok.decode(a).strip() for a in s_ids])
        rec["rouge"].append(r); rec["correct"].append(r > 0.5)
        rec["greedy_lp"].append(glp[0]); rec["greedy_len"].append(gln[0])
        rec["samp_lp"].append(lp); rec["samp_len"].append(ln); rec["emb"].append(emb)
        if qi % 10 == 0 or qi == len(qs) - 1:
            el = time.time() - t0
            acc = np.mean(rec["correct"])
            print(f"[{qi+1}/{len(qs)}] {el/(qi+1):.1f}s/q  acc={acc:.2f}  q={q[:50]!r} greedy={greedy!r}", flush=True)
            save(args, rec)
    save(args, rec)


def save(args, rec):
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, emb=np.stack(rec["emb"]), samp_lp=np.stack(rec["samp_lp"]),
                        samp_len=np.stack(rec["samp_len"]), greedy_lp=np.array(rec["greedy_lp"]),
                        greedy_len=np.array(rec["greedy_len"]), rouge=np.array(rec["rouge"]),
                        correct=np.array(rec["correct"]))
    out.with_suffix(".json").write_text(json.dumps(dict(args=vars(args), question=rec["question"], refs=rec["refs"],
                                                        greedy=rec["greedy"], samples=rec["samples"]), indent=1))


if __name__ == "__main__":
    main()
