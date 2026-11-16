"""A detector of changes of possession, learned on the 7 DFL matches (thousands of changes).

    .venv/bin/python scripts/dfl_change.py [--target touch|sandwich2|events]

Inputs: the direction-free change signals of scripts/change_detector.py
(change_signals(..., vision=False)) plus the frame features, computed on the noisy DFL copies
(cache/dfl_feat_*.pkl). Target: a change between the teams within +-6 frames (positives) against
live frames at least 25 frames from any change (negatives), with DFL's possession as `--target`
defines it (possession/dfl.labels_variant). Gradient boosting.

Reports, on the official public window (never trained on): how well it separates the window's
changes from calm live frames (AUC). Writes cache/dfl_pchange_<target>.npy for our match (read by
scripts/make_submission.py).
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from possession import data, dfl, features  # noqa: E402
from dfl_pretrain import DROP, dfl_features, our_missing_mask  # noqa: E402
import change_detector as cd  # noqa: E402


def targets(lab):
    """(positives, negatives) masks from a label array (0 H, 1 A, 2 D, 3 X)."""
    n = len(lab)
    change = np.zeros(n, bool)
    idx = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    for i in idx:
        if {int(lab[i - 1]), int(lab[i])} == {0, 1}:
            change[i] = True
    near = np.zeros(n, bool)
    far_ok = np.ones(n, bool)
    for i in np.flatnonzero(change):
        near[max(0, i - cd.NEAR):i + cd.NEAR + 1] = True
        far_ok[max(0, i - cd.FAR):i + cd.FAR + 1] = False
    live = np.isin(lab, (0, 1))
    return live & near, live & far_ok


def inputs(df, cols):
    S = cd.change_signals(df, vision=False)
    return np.hstack([S.to_numpy(np.float32), df[cols].to_numpy(np.float32)])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", default="touch")
    p.add_argument("--copies", type=int, default=2)
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    miss = our_missing_mask()
    ours, _ = features.build()
    cols = [c for c in ours.columns if not re.search(DROP, c)]
    Xs, ys = [], []
    for mid in dfl.MATCHES:
        m = dfl.load(mid)
        lab = dfl.labels_variant(mid, m, args.target)
        pos, neg = targets(lab)
        keep = np.flatnonzero(pos | neg)
        keep = keep[(keep % 2 == 0) | pos[keep]]          # thin negatives, keep every positive
        for seed in range(args.copies):
            d = dfl_features(mid, seed, miss)
            Xs.append(inputs(d, cols)[keep])
            ys.append(pos[keep].astype(int))
    X, y = np.vstack(Xs), np.concatenate(ys)
    print("DFL: {} positive frames, {} negative".format(int(y.sum()), int(len(y) - y.sum())))
    det = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=300,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         class_weight="balanced", early_stopping=False,
                                         random_state=0).fit(X, y)
    import joblib
    joblib.dump((det, cols), os.path.join(data.CACHE, "dfl_change_{}.joblib".format(args.target)))
    pc = det.predict_proba(inputs(ours, cols))[:, 1].astype(np.float32)
    out = "dfl_pchange_{}.npy".format(args.target)
    np.save(os.path.join(data.CACHE, out), pc)
    print("wrote cache/" + out)

    # how well it finds the official window's changes (never trained on them)
    gt = data.load_public_gt()
    code = np.select([gt == "H", gt == "A", gt == "D"], [0, 1, 2], 3)
    pos, neg = targets(code)
    t = pos | neg
    print("official window: changes vs calm live frames, AUC {:.2f}".format(
        roc_auc_score(pos[t], pc[t])))


if __name__ == "__main__":
    main()
