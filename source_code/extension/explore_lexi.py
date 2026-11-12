"""Dev exploration (CoQA only): lexicographic scoring, i.e. text agreement with ties broken by a secondary signal."""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from analyze_extension import load_run, all_scores, auroc, paired_bootstrap
prefix = sys.argv[1]
assert Path(prefix).name.endswith("coqa"), "dev exploration is CoQA-only"
run = load_run(prefix)
S, raw, lexsim, _ = all_scores(run, list(range(8, 25)), 16)
y = (~run["correct"].astype(bool)).astype(int)
lex = S["Lexical Similarity"]
def lexi(primary, secondary):
    """Order by primary; break exact ties by secondary (scaled into (0, gap/2))."""
    p = rankdata(primary, method="dense").astype(float)
    s = (rankdata(secondary) - 0.5) / len(secondary)          # in (0, 1)
    return p + 0.5 * s
# within-tie diagnostic: AUROC of each secondary among pairs tied on lex
grp = np.unique(np.round(lex, 9), return_inverse=True)[1]
def within_tie_auc(sec):
    num = den = 0.0
    for g in np.unique(grp):
        m = grp == g; yy, ss = y[m], sec[m]
        pos, neg = ss[yy == 1], ss[yy == 0]
        if len(pos) and len(neg):
            diff = pos[:, None] - neg[None, :]
            num += (diff > 0).sum() + 0.5 * (diff == 0).sum(); den += diff.size
    return num / den, den
tot_pairs = y.sum() * (1 - y).sum()
secs = {k: S[k] for k in ("LN-Entropy", "Perplexity", "EigenScore", "M1 MLES", "M3 HYB (MLES+LNE)")}
print(f"incorrect={y.sum()}, pairs tied on LexSim: {within_tie_auc(secs['Perplexity'])[1]/tot_pairs:.1%} of all pos-neg pairs\n")
print(f"{'LexSim, ties broken by':26s} within-tie AUC   AUROC   vs LexSim [95% CI]      vs ES [95% CI]")
for k, v in secs.items():
    a, _ = within_tie_auc(v)
    sc = lexi(lex, v)
    u = paired_bootstrap(y, sc, lex, n_boot=2000); t = paired_bootstrap(y, sc, S["EigenScore"], n_boot=2000)
    print(f"{k:26s} {a:9.3f}       {100*auroc(y, sc):5.1f}   {100*u['delta']:+.1f} [{100*u['lo']:+.1f},{100*u['hi']:+.1f}]"
          f"     {100*t['delta']:+.1f} [{100*t['lo']:+.1f},{100*t['hi']:+.1f}]")
