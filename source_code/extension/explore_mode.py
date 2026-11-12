"""Dev exploration (CoQA only): spectral mode mass, lambda_max(W)/K, for several kernels W.

W is a K x K similarity matrix between the sampled answers, with W_ii = 1. Its top
eigenvalue divided by K is a smooth estimate of the share of samples in the dominant
meaning (Perron-Frobenius: a block of m identical answers contributes eigenvalue m).
Hallucination score = 1 - lambda_max / K.

    python explore_mode.py results/llama2_coqa
"""
import json, sys
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from scipy.stats import rankdata

from analyze_extension import load_run, all_scores, auroc, paired_bootstrap

prefix = sys.argv[1]
assert Path(prefix).name.endswith("coqa"), "dev exploration is CoQA-only: TriviaQA/NQ are held out"
run = load_run(prefix)
L = run["emb"].shape[2] - 1
band = list(range(L // 4, 3 * L // 4 + 1))
S, raw, lexsim, _ = all_scores(run, band, 16)
y = (~run["correct"].astype(bool)).astype(int)
sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
n, K = run["n"], run["emb"].shape[1]


def lex_kernel(ss):
    W = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            W[i, j] = W[j, i] = 1.0 if ss[i] == ss[j] else sc.score(ss[i], ss[j])["rougeL"].fmeasure
    return W


def emb_cos(X):               # X: (K, d) -> cosine similarity of feature-centred embeddings
    Xc = X - X.mean(-1, keepdims=True)
    Xn = Xc / np.maximum(np.linalg.norm(Xc, axis=-1, keepdims=True), 1e-12)
    return np.clip(Xn @ Xn.T, -1, 1)


def mode_mass(W):
    return float(np.linalg.eigvalsh((W + W.T) / 2)[-1] / K)


Wlex = [lex_kernel(ss) for ss in run["samples"]]
mm = {"MM lex": [], "MM emb16 cos": [], "MM emb16 rbf": [], "MM embband rbf": [],
      "MM lex+emb16 rbf": [], "MM lex+embband rbf": [], "Emb16 mean cos (neg)": []}
for q in range(n):
    E = np.asarray(run["emb"][q], dtype=np.float64)
    C16 = emb_cos(E[:, 16])
    # RBF on cosine distance; bandwidth = median off-diagonal distance over the dataset is
    # label-free, but per-question median would erase the signal -> use a fixed global scale below
    mm["MM lex"].append(mode_mass(Wlex[q]))
    mm["MM emb16 cos"].append(mode_mass(np.maximum(C16, 0)))
    mm["Emb16 mean cos (neg)"].append(-C16[np.triu_indices(K, 1)].mean())
    mm.setdefault("_d16", []).append(1 - C16)
    mm.setdefault("_dband", []).append(np.mean([1 - emb_cos(E[:, l]) for l in band], axis=0))

# global, label-free bandwidth: median off-diagonal cosine distance over all questions
iu = np.triu_indices(K, 1)
h16 = np.median(np.concatenate([d[iu] for d in mm["_d16"]]))
hb = np.median(np.concatenate([d[iu] for d in mm["_dband"]]))
for q in range(n):
    R16 = np.exp(-mm["_d16"][q] / h16); Rb = np.exp(-mm["_dband"][q] / hb)
    mm["MM emb16 rbf"].append(mode_mass(R16))
    mm["MM embband rbf"].append(mode_mass(Rb))
    mm["MM lex+emb16 rbf"].append(mode_mass((Wlex[q] + R16) / 2))
    mm["MM lex+embband rbf"].append(mode_mass((Wlex[q] + Rb) / 2))
print(f"bandwidths: h16={h16:.3f} hband={hb:.3f}")

X = {k: (1 - np.array(v)) if k.startswith("MM") else np.array(v) for k, v in mm.items() if not k.startswith("_")}
X.update({k: S[k] for k in ("EigenScore", "Lexical Similarity", "LN-Entropy", "M3 HYB (MLES+LNE)")})
r = lambda k: rankdata(X[k])
X["TRI (frozen H2)"] = (r("EigenScore") + r("Lexical Similarity") + r("LN-Entropy")) / 3
X["MM lex+embband + LNE"] = (r("MM lex+embband rbf") + r("LN-Entropy")) / 2
X["MM lex+emb16 + LNE"] = (r("MM lex+emb16 rbf") + r("LN-Entropy")) / 2
X["MM lex + LNE"] = (r("MM lex") + r("LN-Entropy")) / 2

ties = lambda v: 1 - len(np.unique(np.round(v, 9))) / len(v)
print(f"{'method':28s} AUROC  tie%   vs ES [95% CI]          vs LexSim [95% CI]")
for k, v in sorted(X.items(), key=lambda kv: -auroc(y, kv[1])):
    t = paired_bootstrap(y, v, X["EigenScore"], n_boot=2000)
    u = paired_bootstrap(y, v, X["Lexical Similarity"], n_boot=2000)
    print(f"{k:28s} {100*auroc(y, v):5.1f}  {100*ties(v):4.0f}   {100*t['delta']:+5.1f} [{100*t['lo']:+.1f},{100*t['hi']:+.1f}]"
          f"    {100*u['delta']:+5.1f} [{100*u['lo']:+.1f},{100*u['hi']:+.1f}]")
