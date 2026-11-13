"""Export per-question data for the GUI (questions, samples, scores, 2-D sample map, eigenvalues).

    python export_demo_data.py   ->  demo_data.js  (window.DEMO = {...})
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

EXT = Path(__file__).resolve().parents[2] / "source_code" / "extension"
sys.path.insert(0, str(EXT))
from analyze_extension import load_run, lexical_similarity, fill_nan, auroc  # noqa: E402

ALPHA, LAYER = 1e-3, 16
NAMES = {"coqa": "CoQA", "triviaqa": "TriviaQA", "nq": "NQ-Open"}


def es_and_geometry(X):
    """EigenScore at one layer, its eigenvalues, and a 2-D PCA map of the K answers."""
    X = X.astype(np.float64)
    Xc = X - X.mean(-1, keepdims=True)
    C = Xc @ Xc.T / (X.shape[-1] - 1)
    ev = np.clip(np.linalg.eigvalsh(C + ALPHA * np.eye(len(X))), 1e-300, None)[::-1]
    Y = Xc - Xc.mean(0, keepdims=True)                # centre across answers for the map
    U, S, _ = np.linalg.svd(Y, full_matrices=False)
    P = U[:, :2] * S[:2]
    return float(np.log10(ev).mean()), ev, P


def main():
    out = {"datasets": {}}
    for key in ("coqa", "triviaqa", "nq"):
        pre = EXT / "results" / f"llama2_{key}"
        run = load_run(pre)
        meta = json.loads(pre.with_suffix(".json").read_text())
        n = run["n"]
        es, evs, maps = np.zeros(n), [], []
        for q in range(n):
            e, ev, P = es_and_geometry(np.asarray(run["emb"][q][:, LAYER]))
            es[q] = e; evs.append(ev); maps.append(P)
        lex = lexical_similarity(run["samples"])
        with np.errstate(divide="ignore", invalid="ignore"):
            nll = -run["samp_lp"] / np.where(run["samp_len"] > 0, run["samp_len"], np.nan)
        lne, _ = fill_nan(np.nanmean(nll, axis=1))
        s_es, s_ag, s_tc = es, -lex, lne                  # all: higher = more likely hallucinated
        pct = lambda v: (rankdata(v) - 0.5) / len(v)      # risk percentile within the dataset
        tri = (rankdata(s_es) + rankdata(s_ag) + rankdata(s_tc)) / 3
        y = (~run["correct"].astype(bool)).astype(int)
        print(key, "AUROC ES", round(100 * auroc(y, s_es), 1), "TRI", round(100 * auroc(y, tri), 1), flush=True)
        qs = []
        for q in range(n):
            m = maps[q]; scale = float(np.abs(m).max()) or 1.0
            qs.append({
                "q": meta["question"][q][:400], "refs": meta["refs"][q][:6], "greedy": meta["greedy"][q],
                "samples": run["samples"][q], "wrong": int(y[q]), "rouge": round(float(run["rouge"][q]), 3),
                "es": round(float(s_es[q]), 4), "ag": round(float(lex[q]), 4), "tc": round(float(s_tc[q]), 4),
                "p_es": round(float(pct(s_es)[q]), 4), "p_ag": round(float(pct(s_ag)[q]), 4),
                "p_tc": round(float(pct(s_tc)[q]), 4), "p_tri": round(float(pct(tri)[q]), 4),
                "ev": [float(f"{v:.3g}") for v in evs[q]],
                "map": [[round(float(a) / scale, 3), round(float(b) / scale, 3)] for a, b in m],
            })
        out["datasets"][key] = {"name": NAMES[key], "accuracy": float(1 - y.mean()), "questions": qs,
                                # sorted reference distributions for the live demo's percentiles
                                "ref": {"es": sorted(map(float, s_es)), "ag": sorted(map(float, s_ag)),
                                        "tc": sorted(map(float, s_tc)), "tri_mean_pct": sorted(
                                            float(v) for v in (pct(s_es) + pct(s_ag) + pct(s_tc)) / 3)}}
    Path("demo_data.js").write_text("window.DEMO = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print("wrote demo_data.js", round(Path("demo_data.js").stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
