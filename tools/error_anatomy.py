"""Where a prediction loses spellF1 on the official public window.

    .venv/bin/python tools/error_anatomy.py outputs/analysis/cd_vision_b32.csv

1. Every true possession, classed at IoU 0.5: matched / too short to be found (no same-team
   prediction covers half of it) / merged (the same-team prediction runs on past it) / split
   (cut into pieces) / shifted.
2. Oracles, from the saved frame probabilities (.npz next to the CSV): what the score would be with
   perfect dead-ball detection, with perfect change times, and against a truth without its
   short possessions. They say which part of the problem is worth working on.
Scores only, nothing from the match is shown.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
from possession import data, decode  # noqa: E402
import possession_metrics as pm  # noqa: E402

def truth():
    """The official ground truth over all frames, X outside the public window."""
    return pm.label_codes(data.load_public_gt())


def classify(gt, pred):
    runs = pm.valid_runs(gt)
    g = pm.possession_spells(gt, runs)
    p = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
    ov = pm.spell_overlaps(g, p)
    hit = pm.matched_gt_flags(ov, len(g), 0.5)
    cls = []
    for i, (s, e, k) in enumerate(g):
        L = e - s
        if hit[i]:
            cls.append(("matched", L))
            continue
        same = [(min(e, pe) - max(s, ps), ps, pe) for ps, pe, pk in p
                if pk == k and pe > s and ps < e]
        cover = sum(o for o, _, _ in same)
        if cover < 0.5 * L:
            cls.append(("missed", L))
        elif len(same) > 1:
            cls.append(("split", L))
        else:
            _, ps, pe = same[0]
            out = max(0, s - ps) + max(0, pe - e)
            cls.append(("merged" if out > 0.5 * L else "shifted", L))
    return cls, len(p)


def score(gt, pred):
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["n_pred_spells"]


def decode_with(proba, mask, team_cost=64.0, dead_cost=128.0):
    out = np.full(len(proba), pm.X, np.int8)
    for s, e in mask:
        out[s:e] = decode.viterbi(proba[s:e], team_cost, dead_cost)
    return out


def main():
    path = sys.argv[1]
    gt = truth()
    pred = pm.label_codes(pm.read_label_csv(path))
    pred = np.where(gt == pm.X, pm.X, pred).astype(np.int8)
    cls, n_pred = classify(gt, pred)
    df = pd.DataFrame(cls, columns=["outcome", "frames"])
    df["short"] = df["frames"] < 2 * data.FPS
    print("true possessions: {}   predicted: {}   spellF1 {:.3f}".format(
        len(df), n_pred, score(gt, pred)[0]))
    print(pd.crosstab(df["outcome"], df["short"].map({True: "< 2 s", False: ">= 2 s"}),
                      margins=True).to_string())

    npz = os.path.splitext(path)[0] + ".npz"
    if not os.path.exists(npz):
        return
    proba = np.load(npz)["proba"]
    runs = [(s, e) for s, e in pm.valid_runs(gt)]
    print("\noracles (fixed costs 64/128):")
    print("  as decoded                               {:.3f} ({})".format(*score(
        gt, decode_with(proba, runs))))
    # perfect dead-ball detection
    q = proba.copy()
    dead = gt == pm.D
    alive = q[:, :2] / np.maximum(q[:, :2].sum(1, keepdims=True), 1e-9)
    q[:, :2] = np.where(dead[:, None], 0.0, alive)
    q[:, 2] = dead.astype(float)
    print("  perfect dead ball                        {:.3f} ({})".format(*score(
        gt, decode_with(np.clip(q, 1e-6, 1), runs))))
    # perfect change times: true segments, team from the mean frame probabilities
    o = np.full(len(gt), pm.X, np.int8)
    for s, e, k in pm.segments(gt, runs):
        o[s:e] = pm.D if k == pm.D else int(np.argmax(proba[s:e, :2].mean(0)))
    print("  perfect change times, model picks team   {:.3f} ({})".format(*score(gt, o)))
    # perfect teams: true team on every live predicted frame
    dec = decode_with(proba, runs)
    o = dec.copy()
    live = (dec != pm.D) & (gt != pm.X)
    o[live & (gt != pm.D)] = gt[live & (gt != pm.D)]
    print("  perfect team on predicted live frames    {:.3f} ({})".format(*score(gt, o)))
    # truth without short possessions (merged into the previous label)
    for lim in (25, 50):
        g2 = gt.copy()
        for s, e, k in pm.possession_spells(gt, runs):
            if e - s < lim and s > 0 and g2[s - 1] != pm.X:
                g2[s:e] = g2[s - 1]
        print("  truth without possessions < {:.0f} s       {:.3f} ({} true)".format(
            lim / data.FPS, score(g2, dec)[0], len(pm.possession_spells(g2, runs))))


if __name__ == "__main__":
    main()
