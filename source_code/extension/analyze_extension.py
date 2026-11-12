"""Offline analysis for the EigenScore extension study (see PLAN.md).

Every detector is computed from the same saved samples, so all comparisons are
paired. Label 1 = incorrect (greedy answer ROUGE-L <= 0.5), and every score is
oriented so that higher = more likely hallucinated.

    python analyze_extension.py results/llama2_coqa results/llama2_triviaqa results/llama2_nq
    python analyze_extension.py --selftest smoke/coqa

Outputs go to --out_dir (default analysis/): summary.md, results.json,
per_layer_auroc.csv and per_layer_auroc.png.
"""
import argparse, itertools, json
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ALPHA = 1e-3          # as in func/metric.py:getEigenIndicator_v0
N_BOOT = 10_000
BOOT_SEED = 2023
FPR = 0.20            # operating point for the self-consistent analysis
SC_LEXSIM = 0.9       # "self-consistent": samples agree at mean pairwise ROUGE-L >= 0.9
M2B_WARMUP = 1        # M2b needs at least this many previous questions


# --------------------------------------------------------------------------- data

def load_run(prefix):
    prefix = Path(prefix)
    sc = np.load(prefix.with_suffix(".npz"))
    meta = json.loads(prefix.with_suffix(".json").read_text())
    n = len(sc["correct"])                       # questions completed (partial runs ok)
    emb = np.load(prefix.with_suffix(".emb.npy"), mmap_mode="r")
    return dict(name=prefix.name, n=n, emb=emb, samples=meta["samples"][:n],
                **{k: sc[k][:n] for k in sc.files})


# --------------------------------------------------------------------- EigenScore

def eigenscore_layers(X, alpha=ALPHA):
    """X: (K, L, d) embeddings of one question -> ES(l) for every layer, shape (L,).

    Same definition as getEigenIndicator_v0: K x K covariance over the d features
    (torch.cov with samples as rows), + alpha*I, mean log10 of the eigenvalues.
    """
    X = X.astype(np.float64)
    Xc = X - X.mean(-1, keepdims=True)
    C = np.einsum("kld,jld->lkj", Xc, Xc) / (X.shape[-1] - 1)
    ev = np.linalg.eigvalsh(C + alpha * np.eye(X.shape[0]))
    return np.log10(np.clip(ev, 1e-300, None)).mean(-1)


def cross_depth_instability(X, band):
    """X1 (exploratory, PLAN_EXPLORATORY.md): 1 - mean RSA correlation between the
    answer-to-answer distance structure of consecutive layers inside the band.

    At each layer l, D_l holds the K(K-1)/2 cosine distances between the K sampled
    answers (embeddings feature-centred as in EigenScore). Representational
    similarity analysis (Kriegeskorte et al., 2008) correlates D_l with D_{l+1}:
    it asks whether the *arrangement* of the answers (which ones resemble which)
    is preserved from one layer to the next, independently of how spread out they
    are, which is what EigenScore measures. A layer where all K answers coincide
    has no arrangement; that layer pair counts as perfectly stable.
    """
    X = X[:, band[0]:band[-1] + 1].astype(np.float64)
    K = X.shape[0]
    Xc = X - X.mean(-1, keepdims=True)
    Xn = Xc / np.maximum(np.linalg.norm(Xc, axis=-1, keepdims=True), 1e-12)
    iu = np.triu_indices(K, 1)
    D = 1.0 - np.einsum("kld,jld->lkj", Xn, Xn)[:, iu[0], iu[1]]      # (layers, K(K-1)/2)
    D = np.maximum(D, 0.0)
    corr = []
    for a in range(len(D) - 1):
        u, v = D[a], D[a + 1]
        if u.std() < 1e-9 or v.std() < 1e-9:           # no arrangement at one of the layers
            corr.append(1.0)
        else:
            corr.append(np.corrcoef(u, v)[0, 1])
    return 1.0 - float(np.mean(corr))


def embedding_scores(run, band):
    """Per question and layer: ES on raw (M0), L2-normalised (M2a) and
    standardised-with-previous-questions (M2b) embeddings."""
    n, K, L, d = run["n"], *run["emb"].shape[1:]
    raw, l2, std = (np.full((n, L), np.nan) for _ in range(3))
    xds = np.full(n, np.nan)
    s1 = np.zeros((L, d)); s2 = np.zeros((L, d)); cnt = 0
    for q in range(n):
        X = np.asarray(run["emb"][q], dtype=np.float64)          # (K, L, d)
        raw[q] = eigenscore_layers(X)
        xds[q] = cross_depth_instability(X, band)
        l2[q] = eigenscore_layers(X / np.linalg.norm(X, axis=-1, keepdims=True))
        if cnt >= M2B_WARMUP:                                     # previous questions only
            mu = s1 / cnt
            sd = np.sqrt(np.maximum(s2 / cnt - mu ** 2, 0)) + 1e-6
            std[q] = eigenscore_layers((X - mu) / sd)
        s1 += X.sum(0); s2 += (X ** 2).sum(0); cnt += K
    return raw, l2, std, xds


# ----------------------------------------------------------------------- baselines

def lexical_similarity(samples):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    out = np.zeros(len(samples))
    for q, ss in enumerate(samples):
        pairs = list(itertools.combinations(ss, 2))
        out[q] = np.mean([1.0 if a == b else scorer.score(a, b)["rougeL"].fmeasure
                          for a, b in pairs]) if pairs else 1.0
    return out


def depth_slope(raw, band):
    """X2: least-squares slope of ES(l) against l over the band (one value per question)."""
    l = np.asarray(band, dtype=float)
    E = raw[:, band]
    return ((E - E.mean(1, keepdims=True)) * (l - l.mean())).sum(1) / ((l - l.mean()) ** 2).sum()


def fill_nan(x):
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.nanmedian(x)
    return x, int(bad.sum())


def all_scores(run, band, ref_layer):
    raw, l2, std, xds = embedding_scores(run, band)
    b = list(band)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppl = -run["greedy_lp"] / np.where(run["greedy_len"] > 0, run["greedy_len"], np.nan)
        nll = -run["samp_lp"] / np.where(run["samp_len"] > 0, run["samp_len"], np.nan)
    lne = np.nanmean(np.where(np.isfinite(nll), nll, np.nan), axis=1) if nll.size else nll
    lexsim = lexical_similarity(run["samples"])

    notes = {}
    ppl, notes["perplexity_imputed"] = fill_nan(ppl)
    lne, notes["ln_entropy_imputed"] = fill_nan(lne)
    mles = raw[:, b].mean(1)
    S = {
        "Perplexity": ppl,
        "LN-Entropy": lne,
        "Lexical Similarity": -lexsim,
        "EigenScore": raw[:, ref_layer],
        "M1 MLES": mles,
        "M2a L2 (single)": l2[:, ref_layer],
        "M2a L2 (multi)": l2[:, b].mean(1),
        "M2b std (single)": std[:, ref_layer],
        "M2b std (multi)": std[:, b].mean(1),
        "M3 HYB (MLES+LNE)": (rankdata(mles) + rankdata(lne)) / 2,
        "M3b MLES+PPL": (rankdata(mles) + rankdata(ppl)) / 2,
        # exploratory, pre-registered in PLAN_EXPLORATORY.md
        "X1 XDS (cross-depth instability)": xds,
        "X2 depth slope": depth_slope(raw, b),
        "X3 MLES+XDS": (rankdata(mles) + rankdata(xds)) / 2,
    }
    return S, raw, lexsim, notes


# ------------------------------------------------------------------------- metrics

def auroc(y, s):
    m = np.isfinite(s)
    y, s = y[m], s[m]
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def auroc_batch(Y, S):
    """Row-wise AUROC via the rank-sum identity (ties -> average ranks). Y, S: (B, n)."""
    R = rankdata(S, axis=1)
    n1 = Y.sum(1); n0 = Y.shape[1] - n1
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((R * Y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def paired_bootstrap(y, a, b, n_boot=N_BOOT, seed=BOOT_SEED, chunk=1000):
    """AUROC(a) - AUROC(b) resampling questions with replacement (same indices for both)."""
    m = np.isfinite(a) & np.isfinite(b)
    y, a, b = y[m], a[m], b[m]
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, len(y), size=(min(chunk, n_boot - start), len(y)))
        Y = y[idx]
        deltas.append(auroc_batch(Y, a[idx]) - auroc_batch(Y, b[idx]))
    d = np.concatenate(deltas)
    d = d[np.isfinite(d)]
    if d.size == 0:                                  # one class only: AUROC undefined
        return dict(delta=np.nan, lo=np.nan, hi=np.nan, p_pos=np.nan, n=int(m.sum()))
    return dict(delta=float(auroc(y, a) - auroc(y, b)), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), p_pos=float((d > 0).mean()), n=int(m.sum()))


def flag_rate_at_fpr(y, s, subset, fpr=FPR):
    """Fraction of `subset` flagged at the threshold giving `fpr` on correct answers."""
    neg = s[(y == 0) & np.isfinite(s)]
    if not subset.any() or neg.size == 0:
        return np.nan
    return float((s[subset] > np.quantile(neg, 1 - fpr)).mean())


# ------------------------------------------------------------------------- figures

def plot_per_layer(per_layer, band, ref_layer, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, muted, grid, base, surf = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated categorical slots 1-3
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": base, "axes.labelcolor": ink2,
                         "xtick.color": muted, "ytick.color": muted})
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    fig.patch.set_facecolor(surf); ax.set_facecolor(surf)
    ax.axvspan(band[0], band[-1], color=grid, alpha=0.6, lw=0, zorder=0)
    ax.text(band[0] + 0.3, 0.985, f"MLES band ({band[0]}–{band[-1]})",
            transform=ax.get_xaxis_transform(), ha="left", va="top", color=ink2, fontsize=9)
    ax.axvline(ref_layer, color=muted, lw=1, ls="--", zorder=1)
    ax.text(ref_layer + 0.3, 0.02, f"original EigenScore\n(index {ref_layer})",
            transform=ax.get_xaxis_transform(), color=ink2, fontsize=8, va="bottom")
    ax.grid(axis="y", color=grid, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    labels = []
    pretty = {"coqa": "CoQA", "triviaqa": "TriviaQA", "nq": "NQ-Open"}
    for (name, auc), c in zip(per_layer.items(), colors):
        name = pretty.get(name.split("_")[-1], name)
        x = np.arange(len(auc))
        ax.plot(x, 100 * auc, color=c, lw=2, marker="o", ms=3.2, label=name, zorder=3,
                markeredgecolor=surf, markeredgewidth=0.8)
        labels.append((100 * auc[-1], name))
    # direct labels at line ends, nudged apart, in text ink (identity carried by the legend + marker)
    labels.sort()
    placed = []
    for yv, name in labels:
        while placed and yv - placed[-1] < 1.6:
            yv = placed[-1] + 1.6
        placed.append(yv)
        ax.text(len(auc) - 0.4, yv, name, color=ink2, fontsize=8.5, va="center")
    ax.set_xlim(-0.5, len(auc) + 3.5)
    ax.set_xticks(range(0, len(auc), 4))
    ax.set_xlabel("Hidden-state index (0 = embeddings, 32 = last layer)")
    ax.set_ylabel("AUROC (%)")
    ax.set_title("Single-layer EigenScore AUROC by layer", color=ink, fontsize=11, loc="left", pad=22)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8.5,
              labelcolor=ink2, borderaxespad=0.2)
    fig.tight_layout(); fig.savefig(path, facecolor=surf); plt.close(fig)


# ---------------------------------------------------------------------------- main

def analyse(prefixes, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report, per_layer = {}, {}
    for p in prefixes:
        run = load_run(p)
        L = run["emb"].shape[2] - 1                           # decoder layers (32)
        band = list(range(L // 4, 3 * L // 4 + 1))            # pre-registered: 8..24
        ref = int((L + 1) / 2)                                # int(len(hidden_states)/2) = 16
        y = (~run["correct"].astype(bool)).astype(int)
        S, raw, lexsim, notes = all_scores(run, band, ref)

        aucs = {k: auroc(y, v) for k, v in S.items()}
        layer_auc = np.array([auroc(y, raw[:, l]) for l in range(raw.shape[1])])
        per_layer[run["name"]] = layer_auc
        tests = {k: paired_bootstrap(y, S[k], S["EigenScore"])
                 for k in S if k != "EigenScore"}
        tests["M3 HYB vs M1 MLES"] = paired_bootstrap(y, S["M3 HYB (MLES+LNE)"], S["M1 MLES"])
        tests["X3 MLES+XDS vs M1 MLES"] = paired_bootstrap(y, S["X3 MLES+XDS"], S["M1 MLES"])
        from scipy.stats import spearmanr
        overlap = dict(spearman_MLES_XDS=float(spearmanr(S["M1 MLES"], S["X1 XDS (cross-depth instability)"])[0]),
                       spearman_ES_XDS=float(spearmanr(S["EigenScore"], S["X1 XDS (cross-depth instability)"])[0]))

        sc = (y == 1) & (lexsim >= SC_LEXSIM)
        flags = {k: flag_rate_at_fpr(y, v, sc) for k, v in S.items()}
        best = int(np.nanargmax(layer_auc)) if np.isfinite(layer_auc).any() else 0
        report[run["name"]] = dict(
            n=run["n"], accuracy=float(1 - y.mean()), band=[band[0], band[-1]], ref_layer=ref,
            auroc=aucs, tests=tests, per_layer_auroc=layer_auc.tolist(),
            best_layer=dict(index=best, auroc=float(layer_auc[best])),
            self_consistent=dict(n=int(sc.sum()), n_incorrect=int(y.sum()), flag_rate_at_20fpr=flags),
            notes=notes, xds_overlap=overlap)

    (out_dir / "results.json").write_text(json.dumps(report, indent=1))
    names = list(per_layer)
    with open(out_dir / "per_layer_auroc.csv", "w") as f:
        f.write("layer," + ",".join(names) + "\n")
        for l in range(len(per_layer[names[0]])):
            f.write(f"{l}," + ",".join(f"{per_layer[n][l]:.4f}" for n in names) + "\n")
    plot_per_layer(per_layer, report[names[0]]["band"], report[names[0]]["ref_layer"],
                   out_dir / "per_layer_auroc.png")
    write_summary(report, out_dir / "summary.md")
    return report


def write_summary(report, path):
    names = list(report)
    methods = list(report[names[0]]["auroc"])
    pc = lambda x: "–" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"
    lines = ["# Extension results (auto-generated by analyze_extension.py)", "",
             "| dataset | n | accuracy |", "|---|---|---|"]
    lines += [f"| {d} | {r['n']} | {100 * r['accuracy']:.1f}% |" for d, r in report.items()]
    lines += ["", "## AUROC (%), label = ROUGE-L <= 0.5", "",
              "| method | " + " | ".join(names) + " | mean | sd |", "|---" * (len(names) + 3) + "|"]
    for m in methods:
        v = np.array([report[d]["auroc"][m] for d in names], dtype=float)
        lines.append(f"| {m} | " + " | ".join(pc(x) for x in v) +
                     f" | {pc(np.nanmean(v))} | {100 * np.nanstd(v):.1f} |")
    lines += ["", "## Paired bootstrap vs original EigenScore (10,000 resamples)", "",
              "Delta AUROC in points, 95% CI, P(delta > 0). **Primary: M1 MLES.** Others secondary.", ""]
    lines += ["| comparison | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for k in report[names[0]]["tests"]:
        cells = []
        for d in names:
            t = report[d]["tests"][k]
            cells.append(f"{100 * t['delta']:+.1f} [{100 * t['lo']:+.1f}, {100 * t['hi']:+.1f}] p={t['p_pos']:.2f}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "## Layer robustness", "", "| dataset | MLES | EigenScore (idx 16) | best single layer (hindsight) |",
              "|---|---|---|---|"]
    for d, r in report.items():
        lines.append(f"| {d} | {pc(r['auroc']['M1 MLES'])} | {pc(r['auroc']['EigenScore'])} | "
                     f"{pc(r['best_layer']['auroc'])} (idx {r['best_layer']['index']}) |")
    lines += ["", "## Exploratory: overlap of XDS with volume-based scores (Spearman)", "",
              "| dataset | rho(MLES, XDS) | rho(EigenScore, XDS) |", "|---|---|---|"]
    for d, r in report.items():
        o = r["xds_overlap"]
        lines.append(f"| {d} | {o['spearman_MLES_XDS']:.2f} | {o['spearman_ES_XDS']:.2f} |")
    lines += ["", f"## Self-consistent hallucinations (incorrect, mean pairwise ROUGE-L >= {SC_LEXSIM})", "",
              "Fraction flagged at the threshold giving 20% FPR on correct answers.", "",
              "| method | " + " | ".join(f"{d} (n={report[d]['self_consistent']['n']})" for d in names) + " |",
              "|---" * (len(names) + 1) + "|"]
    for m in methods:
        lines.append(f"| {m} | " + " | ".join(pc(report[d]['self_consistent']['flag_rate_at_20fpr'][m])
                                              for d in names) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def selftest(prefix):
    """ES implementation == repo's getEigenScoreFromEmbeddings; bootstrap AUROC == sklearn."""
    import sys, torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from func.metric import getEigenScoreFromEmbeddings
    run = load_run(prefix)
    worst = 0.0
    for q in range(run["n"]):
        X = np.asarray(run["emb"][q], dtype=np.float64)
        ours = eigenscore_layers(X)
        for l in (8, 16, 24):
            ref, _ = getEigenScoreFromEmbeddings(torch.tensor(X[:, l]), alpha=ALPHA)
            worst = max(worst, abs(ours[l] - ref))
    print(f"[{'ok' if worst < 1e-6 else 'FAIL'}] EigenScore matches func/metric.py (max |diff| {worst:.2e})")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300); s = rng.normal(size=300) + y * 0.5; s[:30] = np.round(s[:30])
    a1, a2 = roc_auc_score(y, s), auroc_batch(y[None], s[None])[0]
    print(f"[{'ok' if abs(a1 - a2) < 1e-12 else 'FAIL'}] rank-sum AUROC == sklearn ({a1:.6f} vs {a2:.6f})")
    rng2 = np.random.default_rng(1)
    base = rng2.normal(size=(10, 1, 64)) * 3 + rng2.normal(size=(1, 1, 64))
    same = np.repeat(base, 33, axis=1)                                   # identical geometry at every layer
    shuf = np.stack([base[rng2.permutation(10), 0] for _ in range(33)], 1)  # samples reshuffled per layer
    dup = np.repeat(np.repeat(base[:1], 10, 0), 33, 1)                   # all K answers identical
    b_ = list(range(8, 25))
    x0, x1, x2 = (cross_depth_instability(z, b_) for z in (same, shuf, dup))
    ok = abs(x0) < 1e-9 and x1 > 0.2 and x2 == 0.0
    print(f"[{'ok' if ok else 'FAIL'}] XDS: stable geometry {x0:.1e}, reshuffled {x1:.2f}, all-duplicate {x2:.1f}")
    t = paired_bootstrap(y, s, s)
    print(f"[{'ok' if t['delta'] == 0 and t['lo'] == 0 == t['hi'] else 'FAIL'}] bootstrap of a method vs itself is 0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--out_dir", default="analysis")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.prefixes[0])
    else:
        analyse(args.prefixes, args.out_dir)
