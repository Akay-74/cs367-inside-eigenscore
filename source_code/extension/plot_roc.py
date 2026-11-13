"""ROC curves of TRI vs its three input signals on all three datasets (paper figure).

    python plot_roc.py results/llama2_coqa results/llama2_triviaqa results/llama2_nq out.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve

from analyze_extension import load_run, all_scores, auroc

# fixed categorical order: TRI first (the method under study), then its inputs
SERIES = [("TRI (ours)", "#2a78d6", 2.0), ("EigenScore", "#eb6834", 1.3),
          ("Text agreement", "#1baf7a", 1.3), ("LN-Entropy", "#eda100", 1.3)]
TITLES = {"coqa": "CoQA (dev)", "triviaqa": "TriviaQA (held-out)", "nq": "NQ-Open (held-out)"}


def main(prefixes, out):
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "font.family": "serif"})
    fig, axes = plt.subplots(1, len(prefixes), figsize=(7.0, 2.45), sharey=True)
    for ax, pre in zip(axes, prefixes):
        run = load_run(pre)
        L = run["emb"].shape[2] - 1
        S, _, _, _ = all_scores(run, list(range(L // 4, 3 * L // 4 + 1)), 16)
        y = (~run["correct"].astype(bool)).astype(int)
        es, lex, lne = S["EigenScore"], S["Lexical Similarity"], S["LN-Entropy"]
        tri = (rankdata(es) + rankdata(lex) + rankdata(lne)) / 3
        for (name, col, lw), s in zip(SERIES, (tri, es, lex, lne)):
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=col, lw=lw, label=f"{name} {100 * auroc(y, s):.1f}",
                    zorder=3 if name.startswith("TRI") else 2)
        ax.plot([0, 1], [0, 1], ls=":", lw=0.6, color="#888888")
        ax.set_title(TITLES[Path(pre).name.replace("llama2_", "")], fontsize=8)
        ax.set_xlabel("False positive rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.grid(lw=0.3, color="#dddddd"); ax.tick_params(width=0.5, length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(loc="lower right", fontsize=6, frameon=False, handlelength=1.4, title="AUROC", title_fontsize=6)
    axes[0].set_ylabel("True positive rate")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
