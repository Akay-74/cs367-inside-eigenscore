"""Development-set exploration (CoQA only). TriviaQA and NQ are held out.

Anything that looks good here must be frozen in PLAN_CONFIRMATORY.md and then tested,
unchanged, on TriviaQA and NQ. This script refuses to read any dataset except CoQA.

    python explore_dev.py results/llama2_coqa
"""
import itertools, json, sys
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from scipy.stats import rankdata, spearmanr

from analyze_extension import load_run, all_scores, auroc, paired_bootstrap

prefix = sys.argv[1]
assert Path(prefix).name.endswith("coqa"), "dev exploration is CoQA-only: TriviaQA/NQ are held out"

run = load_run(prefix)
L = run["emb"].shape[2] - 1
band = list(range(L // 4, 3 * L // 4 + 1))
S, raw, lexsim, _ = all_scores(run, band, 16)
y = (~run["correct"].astype(bool)).astype(int)
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
meta = json.loads(Path(prefix).with_suffix(".json").read_text())
greedy, samples = meta["greedy"][:run["n"]], run["samples"]

# --- extra cheap signals from the saved texts / log-probs (no new generation) ---
def rl(a, b):
    return 1.0 if a == b else scorer.score(a, b)["rougeL"].fmeasure

support = np.array([np.mean([rl(g, s) for s in ss]) for g, ss in zip(greedy, samples)])
def clusters(ss, thr=0.5):          # greedy single-link clustering by ROUGE-L > thr
    lab = [-1] * len(ss); c = 0
    for i in range(len(ss)):
        if lab[i] < 0:
            lab[i] = c
            for j in range(i + 1, len(ss)):
                if lab[j] < 0 and rl(ss[i], ss[j]) > thr:
                    lab[j] = c
            c += 1
    return np.bincount(lab)
def entropy(cnt):
    p = cnt / cnt.sum(); return float(-(p * np.log(p)).sum())
sem_ent = np.array([entropy(clusters(ss)) for ss in samples])
n_distinct = np.array([len(set(s.strip().lower() for s in ss)) for ss in samples])
mean_len = run["samp_len"].mean(1).astype(float)
std_len = run["samp_len"].std(1).astype(float)

X = dict(S)
X.update({
    "Greedy support (neg)": -support,
    "Discrete semantic entropy": sem_ent,
    "# distinct answers": n_distinct.astype(float),
    "Mean answer length": mean_len,
    "Std answer length": std_len,
})
r = lambda k: rankdata(X[k])
fusions = {
    "F: ES + LexSim": ("EigenScore", "Lexical Similarity"),
    "F: ES + LexSim + LNE": ("EigenScore", "Lexical Similarity", "LN-Entropy"),
    "F: MLES + LexSim + LNE": ("M1 MLES", "Lexical Similarity", "LN-Entropy"),
    "F: ES + support": ("EigenScore", "Greedy support (neg)"),
    "F: ES + support + LNE": ("EigenScore", "Greedy support (neg)", "LN-Entropy"),
    "F: ES + LexSim + support + LNE": ("EigenScore", "Lexical Similarity", "Greedy support (neg)", "LN-Entropy"),
    "F: LexSim + LNE": ("Lexical Similarity", "LN-Entropy"),
    "F: ES + SemEnt + LNE": ("EigenScore", "Discrete semantic entropy", "LN-Entropy"),
}
for name, parts in fusions.items():
    X[name] = np.mean([r(p) for p in parts], axis=0)

print(f"CoQA dev: n={run['n']}, incorrect={y.sum()}")
print(f"{'method':38s} AUROC   vs EigenScore [95% CI]        vs LexSim [95% CI]")
for k, v in sorted(X.items(), key=lambda kv: -auroc(y, kv[1])):
    t = paired_bootstrap(y, v, X["EigenScore"], n_boot=2000)
    u = paired_bootstrap(y, v, X["Lexical Similarity"], n_boot=2000)
    print(f"{k:38s} {100*auroc(y, v):5.1f}   {100*t['delta']:+5.1f} [{100*t['lo']:+.1f},{100*t['hi']:+.1f}]"
          f"     {100*u['delta']:+5.1f} [{100*u['lo']:+.1f},{100*u['hi']:+.1f}]")
print("\nSpearman correlations (signal complementarity):")
keys = ["EigenScore", "Lexical Similarity", "LN-Entropy", "Greedy support (neg)", "Perplexity"]
for a, b in itertools.combinations(keys, 2):
    print(f"  {a:22s} ~ {b:22s} rho = {spearmanr(X[a], X[b])[0]:.2f}")
