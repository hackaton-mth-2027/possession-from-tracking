"""Is version B really better than version A? Paired bootstrap over the official possessions.

    .venv/bin/python tools/compare.py A.csv B.csv

Both predictions are scored on the same resampled possessions of the official public window,
so the noise they share cancels: the interval on the DIFFERENCE is much narrower than the one on
each score. Decide on a change only when the 90% interval of the difference excludes 0.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
import possession_metrics as pm  # noqa: E402
from error_anatomy import truth  # noqa: E402


def flags_and_preds(gt, pred, runs):
    """Per true possession and IoU threshold: matched or not; and, per true possession, how many
    predicted possessions overlap it most (to resample predictions with their possessions)."""
    filled = pm.fill_abstentions(pred, runs)
    g = pm.possession_spells(gt, runs)
    p = pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(g, p)
    flags = np.vstack([pm.matched_gt_flags(ov, len(g), t) for t in pm.SPELL_IOUS])
    # each predicted possession is attached to the true live segment it overlaps most (by
    # midpoint), so a draw resamples predictions together with the stretch of match they sit in
    mids = np.array([(a + b) / 2 for a, b, _ in p])
    owner = np.searchsorted(np.array([a for a, _, _ in g]), mids, side="right") - 1
    n_pred_of = np.bincount(np.clip(owner, 0, len(g) - 1), minlength=len(g))
    return flags, n_pred_of


def f1(flags, n_pred_of, picks):
    tp = flags[:, picks].sum(1)
    n_gt, n_pred = len(picks), n_pred_of[picks].sum()
    return np.mean(np.where(tp > 0, 2 * tp / (n_gt + n_pred), 0.0))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--only", help="score only these frames, e.g. 79978-90238,142500-150592")
    args = p.parse_args()
    gt = truth()
    if args.only:
        keep = np.zeros(len(gt), bool)
        for part in args.only.split(","):
            lo, hi = (int(v) for v in part.split("-"))
            keep[lo:hi + 1] = True
        gt = np.where(keep, gt, pm.X).astype(gt.dtype)
    runs = pm.valid_runs(gt)
    preds = []
    for path in (args.a, args.b):
        q = pm.label_codes(pm.read_label_csv(path))
        preds.append(np.where(gt == pm.X, pm.X, q).astype(np.int8))
    fa, na = flags_and_preds(gt, preds[0], runs)
    fb, nb = flags_and_preds(gt, preds[1], runs)
    n = fa.shape[1]
    rng = np.random.RandomState(0)
    diffs = []
    for _ in range(args.draws):
        picks = rng.randint(0, n, n)
        diffs.append(f1(fb, nb, picks) - f1(fa, na, picks))
    diffs = np.array(diffs)
    sa = pm.evaluate(gt, preds[0])["spell_f1_mean"]
    sb = pm.evaluate(gt, preds[1])["spell_f1_mean"]
    lo, hi = np.percentile(diffs, [5, 95])
    print("{} true possessions | A {:.3f}  B {:.3f}  B-A {:+.3f}  90% interval [{:+.3f}, {:+.3f}]"
          "  P(B better) {:.2f}".format(n, sa, sb, sb - sa, lo, hi, np.mean(diffs > 0)))


if __name__ == "__main__":
    main()
