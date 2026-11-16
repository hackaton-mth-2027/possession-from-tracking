"""Where the submission loses spellF1, on the Metrica validation set (1,117 possessions).

    .venv/bin/python scripts/metrica_anatomy.py

1. Every true possession at IoU 0.5 (tools/error_anatomy.classify): matched / missed (no
   same-team prediction covers half of it) / merged / split / shifted, short (< 2 s) or not.
2. Oracles, each replacing one part of the submission by the truth and decoding as the
   submission does (28 / 23 / 32, dead ball from DFL + PFF): perfect dead ball, perfect team
   probabilities (dead ball as predicted), perfect change detector (p_change 1 within +-6 frames of
   a true change, 0 elsewhere), perfect change times (true segments, team from the model), and the
   truth without its short possessions. They say which part is worth working on.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
from possession import data  # noqa: E402
import possession_metrics as pm  # noqa: E402
import change_detector as cd  # noqa: E402
from compare import flags_and_preds, f1  # noqa: E402
from error_anatomy import classify  # noqa: E402
import validate_clean as vc  # noqa: E402
from dead_change import SUBMISSION, submission_samples  # noqa: E402


def pooled(pairs):
    fl, npd = [], []
    for lab, pred in pairs:
        f, n = flags_and_preds(lab, pred.astype(np.int8), [(0, len(lab))])
        fl.append(f)
        npd.append(n)
    fl, npd = np.hstack(fl), np.concatenate(npd)
    return f1(fl, npd, np.arange(fl.shape[1])), int(npd.sum())


def main():
    warnings.filterwarnings("ignore")
    S = submission_samples()
    units = []
    for i, (lab, pp, ps, pc) in enumerate(S.units["Metrica"]):
        units.append((lab, vc.unit_proba(S, "Metrica", i, pp, ps, SUBMISSION), pc))
    dec = lambda p, c: vc.decode_unit(p, c, SUBMISSION)

    base = [(lab, dec(p, c)) for lab, p, c in units]
    rows = []
    for lab, pred in base:
        rows += classify(lab, pred)[0]
    df = pd.DataFrame(rows, columns=["outcome", "frames"])
    df["length"] = np.where(df.frames < 2 * data.FPS, "< 2 s", ">= 2 s")
    sc, n_pred = pooled(base)
    print("submission on Metrica: spellF1 {:.3f}, {} true possessions, {} predicted".format(
        sc, len(df), n_pred))
    print(pd.crosstab(df.outcome, df.length, margins=True).to_string())
    print("true possessions < 2 s: {:.0%}; median length {:.1f} s".format(
        np.mean(df.frames < 2 * data.FPS), df.frames.median() / data.FPS))

    def oracle(name, fn):
        s, n = pooled([(lab, fn(lab, p, c)) for lab, p, c in units])
        print("  {:<52s} {:.3f} ({} predicted)".format(name, s, n), flush=True)

    print("\noracles:")
    oracle("as decoded", lambda lab, p, c: dec(p, c))

    def perfect_dead(lab, p, c):
        q = p.copy()
        alive = q[:, :2] / np.maximum(q[:, :2].sum(1, keepdims=True), 1e-9)
        d = lab == pm.D
        q[:, :2] = np.where(d[:, None], 0.0, alive)
        q[:, 2] = d
        return dec(np.clip(q, 1e-6, 1), c)
    oracle("perfect dead ball", perfect_dead)

    def perfect_team(lab, p, c):
        q = p.copy()
        live = q[:, 0] + q[:, 1]
        q[:, 0] = np.where(lab == pm.H, live, np.where(lab == pm.A, 0.0, q[:, 0]))
        q[:, 1] = np.where(lab == pm.A, live, np.where(lab == pm.H, 0.0, q[:, 1]))
        return dec(np.clip(q, 1e-6, 1), c)
    oracle("perfect team probabilities (dead ball as predicted)", perfect_team)

    def perfect_change(lab, p, c):
        q = np.zeros(len(lab))
        for i in np.flatnonzero(lab[1:] != lab[:-1]) + 1:
            if {int(lab[i - 1]), int(lab[i])} == {pm.H, pm.A}:
                q[max(0, i - cd.NEAR):i + cd.NEAR + 1] = 1.0
        return dec(p, q)
    oracle("perfect change detector", perfect_change)

    def perfect_times(lab, p, c):
        o = np.empty(len(lab), np.int8)
        for s, e, k in pm.segments(lab, [(0, len(lab))]):
            o[s:e] = pm.D if k == pm.D else int(np.argmax(p[s:e, :2].mean(0)))
        return o
    oracle("perfect change times, model picks the team", perfect_times)

    for lim in (1.0, 2.0, 4.0):
        pairs = []
        for (lab, pred) in base:
            g2 = lab.copy()
            for s, e, k in pm.possession_spells(lab, [(0, len(lab))]):
                if e - s < lim * data.FPS and s > 0:
                    g2[s:e] = g2[s - 1]
            pairs.append((g2, pred))
        s, n = pooled(pairs)
        print("  {:<52s} {:.3f}".format("truth without possessions < {:g} s".format(lim), s))


if __name__ == "__main__":
    main()
