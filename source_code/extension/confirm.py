"""Confirmatory test of PLAN_CONFIRMATORY.md on the held-out datasets.

    python confirm.py results/llama2_triviaqa results/llama2_nq   # held-out test
    python confirm.py --dev results/llama2_coqa                    # dry run on the dev set

Writes analysis/CONFIRMATORY_RESULTS.md (or analysis/confirm_dev.md with --dev).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from analyze_extension import load_run, all_scores, auroc, paired_bootstrap

MARGIN = 0.010   # H2b non-inferiority margin (AUROC)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def main(prefixes, dev=False):
    rows, sup = [], []          # sup: (label, dataset, one-sided p) for Holm
    for pre in prefixes:
        run = load_run(pre)
        L = run["emb"].shape[2] - 1
        S, _, _, _ = all_scores(run, list(range(L // 4, 3 * L // 4 + 1)), 16)
        y = (~run["correct"].astype(bool)).astype(int)
        es, lex, lne = S["EigenScore"], S["Lexical Similarity"], S["LN-Entropy"]
        tri = (rankdata(es) + rankdata(lex) + rankdata(lne)) / 3
        name = Path(pre).name.replace("llama2_", "")
        auc = {k: auroc(y, v) for k, v in
               {"ES": es, "LexSim": lex, "LNE": lne, "TRI": tri, "HYB": S["M3 HYB (MLES+LNE)"]}.items()}
        tests = {
            "H1 LexSim > ES": paired_bootstrap(y, lex, es),
            "H2a TRI > ES": paired_bootstrap(y, tri, es),
            "H3 HYB > ES": paired_bootstrap(y, S["M3 HYB (MLES+LNE)"], es),
            "H2b TRI vs LexSim": paired_bootstrap(y, tri, lex),
        }
        rows.append((name, run["n"], 1 - y.mean(), auc, tests))
        for k in ("H1 LexSim > ES", "H2a TRI > ES", "H3 HYB > ES"):
            sup.append((k, name, 1 - tests[k]["p_pos"]))

    adj = holm([s[2] for s in sup])
    pc = lambda x: f"{100 * x:.1f}"
    out = ["# Confirmatory results" + (" (DEV DRY RUN on CoQA, not a confirmation)" if dev else ""),
           "", "Hypotheses frozen in PLAN_CONFIRMATORY.md (commit 4a26da7) before these data were analysed.", "",
           "| dataset | n | accuracy | ES | LexSim | LNE | HYB | **TRI** |", "|---|---|---|---|---|---|---|---|"]
    for name, n, acc, auc, _ in rows:
        out.append(f"| {name} | {n} | {pc(acc)}% | {pc(auc['ES'])} | {pc(auc['LexSim'])} | {pc(auc['LNE'])} | "
                   f"{pc(auc['HYB'])} | **{pc(auc['TRI'])}** |")
    out += ["", "| test | dataset | ΔAUROC [95% CI] | one-sided p | Holm p | holds? |", "|---|---|---|---|---|---|"]
    k = 0
    for name, _, _, _, tests in rows:
        for t in ("H1 LexSim > ES", "H2a TRI > ES", "H3 HYB > ES"):
            r = tests[t]; s = sup[k]; a = adj[k]; k += 1
            out.append(f"| {t} | {name} | {100*r['delta']:+.1f} [{100*r['lo']:+.1f}, {100*r['hi']:+.1f}] | "
                       f"{s[2]:.4f} | {a:.4f} | {'yes' if a < 0.05 else 'no'} |")
        r = tests["H2b TRI vs LexSim"]
        ni = r["lo"] > -MARGIN
        out.append(f"| H2b TRI ≥ LexSim − 1pt | {name} | {100*r['delta']:+.1f} [{100*r['lo']:+.1f}, {100*r['hi']:+.1f}] | "
                   f"– | – | {'yes' if ni else 'no'}{' (strictly better)' if r['lo'] > 0 else ''} |")
    verdict = []
    for t in ("H1 LexSim > ES", "H2a TRI > ES", "H3 HYB > ES"):
        holds = [adj[i] < 0.05 for i, s in enumerate(sup) if s[0] == t]
        verdict.append(f"- **{t}**: " + ("confirmed on all datasets" if all(holds) else
                                         "partial" if any(holds) else "not confirmed"))
    h2b = [r[4]["H2b TRI vs LexSim"]["lo"] > -MARGIN for r in rows]
    verdict.append("- **H2b TRI non-inferior to LexSim**: " + ("confirmed on all datasets" if all(h2b) else
                                                               "partial" if any(h2b) else "not confirmed"))
    out += ["", "## Verdict", ""] + verdict
    dest = Path("analysis") / ("confirm_dev.md" if dev else "CONFIRMATORY_RESULTS.md")
    dest.parent.mkdir(exist_ok=True); dest.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dev"]
    dev = "--dev" in sys.argv
    if not dev:
        assert not any(a.endswith("coqa") for a in args), "CoQA is the dev set; use --dev"
    main(args, dev)
