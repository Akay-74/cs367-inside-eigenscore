"""Dev exploration (CoQA only): probability-weighted mass of the greedy answer's meaning.

PMM = sum of p(s) over distinct sampled strings that agree with the greedy answer
      (ROUGE-L > 0.5), divided by the sum of p(s) over all distinct strings (greedy included).
p(s) = exp(sum of token log-probs) at temperature 1. Score = 1 - PMM.
"""
import json, sys
from pathlib import Path
import numpy as np
from rouge_score import rouge_scorer
from scipy.stats import rankdata
from analyze_extension import load_run, all_scores, auroc, paired_bootstrap
prefix = sys.argv[1]
assert Path(prefix).name.endswith("coqa"), "dev exploration is CoQA-only"
run = load_run(prefix)
S, _, _, _ = all_scores(run, list(range(8, 25)), 16)
y = (~run["correct"].astype(bool)).astype(int)
meta = json.loads(Path(prefix).with_suffix(".json").read_text())
sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rl = lambda a, b: 1.0 if a == b else sc.score(a, b)["rougeL"].fmeasure
pmm, cnt = [], []
for q in range(run["n"]):
    g = meta["greedy"][q]
    probs = {g: np.exp(run["greedy_lp"][q])}
    for s, lp in zip(run["samples"][q], run["samp_lp"][q]):
        probs.setdefault(s, np.exp(lp))
    tot = sum(probs.values())
    pmm.append(sum(p for s, p in probs.items() if rl(g, s) > 0.5) / tot if tot > 0 else 0.0)
    cnt.append(np.mean([rl(g, s) > 0.5 for s in run["samples"][q]]))
lex, lne, es = S["Lexical Similarity"], S["LN-Entropy"], S["EigenScore"]
X = {"EigenScore": es, "Lexical Similarity": lex,
     "CTC (frozen H4)": rankdata(lex, method="dense") + 0.5 * (rankdata(lne) - 0.5) / len(lne),
     "PMM (prob-weighted mode mass)": 1 - np.array(pmm),
     "Count mode mass": 1 - np.array(cnt)}
X["PMM + LexSim"] = (rankdata(X["PMM (prob-weighted mode mass)"]) + rankdata(lex)) / 2
X["PMM, ties by LNE"] = rankdata(X["PMM (prob-weighted mode mass)"], method="dense") + 0.5 * (rankdata(lne) - 0.5) / len(lne)
print(f"{'method':32s} AUROC  vs LexSim [95% CI]     vs CTC [95% CI]")
for k, v in sorted(X.items(), key=lambda kv: -auroc(y, kv[1])):
    u = paired_bootstrap(y, v, lex, n_boot=2000); c = paired_bootstrap(y, v, X["CTC (frozen H4)"], n_boot=2000)
    print(f"{k:32s} {100*auroc(y, v):5.1f}  {100*u['delta']:+.1f} [{100*u['lo']:+.1f},{100*u['hi']:+.1f}]    {100*c['delta']:+.1f} [{100*c['lo']:+.1f},{100*c['hi']:+.1f}]")
