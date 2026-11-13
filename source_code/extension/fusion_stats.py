"""Post-hoc description of the confirmed fusion (TRI) on all three datasets.

Signal correlations and a leave-one-signal-out ablation. This is descriptive, NOT part of
the frozen confirmatory test (PLAN_CONFIRMATORY.md).
    python fusion_stats.py results/llama2_coqa results/llama2_triviaqa results/llama2_nq
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from analyze_extension import load_run, eigenscore_layers, lexical_similarity, fill_nan, auroc, paired_bootstrap

out = {}
for pre in sys.argv[1:]:
    run = load_run(pre); name = Path(pre).name.replace("llama2_", "")
    y = (~run["correct"].astype(bool)).astype(int)
    es = np.array([eigenscore_layers(np.asarray(run["emb"][q][:, 16:17], dtype=np.float64))[0] for q in range(run["n"])])
    lex = -lexical_similarity(run["samples"])
    with np.errstate(divide="ignore", invalid="ignore"):
        nll = -run["samp_lp"] / np.where(run["samp_len"] > 0, run["samp_len"], np.nan)
    lne, _ = fill_nan(np.nanmean(nll, axis=1))
    r = {k: rankdata(v) for k, v in dict(ES=es, LEX=lex, LNE=lne).items()}
    tri = (r["ES"] + r["LEX"] + r["LNE"]) / 3
    abl = {"ES+LEX": (r["ES"] + r["LEX"]) / 2, "ES+LNE": (r["ES"] + r["LNE"]) / 2, "LEX+LNE": (r["LEX"] + r["LNE"]) / 2}
    d = dict(auc={k: auroc(y, v) for k, v in dict(ES=es, LEX=lex, LNE=lne, TRI=tri, **abl).items()},
             rho={f"{a}~{b}": float(spearmanr(r[a], r[b])[0]) for a, b in (("ES", "LEX"), ("ES", "LNE"), ("LEX", "LNE"))},
             tri_vs={k: paired_bootstrap(y, tri, v) for k, v in dict(ES=es, LEX=lex, LNE=lne, **abl).items()})
    out[name] = d
    print(name, {k: round(100 * v, 1) for k, v in d["auc"].items()}, {k: round(v, 2) for k, v in d["rho"].items()}, flush=True)
    print("   TRI minus:", {k: f"{100*t['delta']:+.1f} [{100*t['lo']:+.1f},{100*t['hi']:+.1f}]" for k, t in d["tri_vs"].items()}, flush=True)
Path("analysis/fusion_stats.json").write_text(json.dumps(out, indent=1))
