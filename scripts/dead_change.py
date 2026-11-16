"""A detector of dead-ball boundaries (the ball going dead, play restarting) that lowers the
decoder's dead-ball cost where one is likely, as the change detector does for the team.

    .venv/bin/python scripts/dead_change.py            # boundary detector
    .venv/bin/python scripts/dead_change.py team       # a few DFL + PFF models in the team bag
    .venv/bin/python scripts/dead_change.py choice     # dead-ball source x boundary detector
    .venv/bin/python scripts/dead_change.py proba      # the submission's boundary probabilities

Targets: frames within +-6 of a live <-> dead transition against graded frames at least 25 from
any (scripts/dfl_change.py's thresholds), on DFL (ball status), DFL + PFF or PFF alone (events:
ball out, or end of the last action before a set piece, until the restart). Inputs: the change
detector's (direction-free change signals + frame features) plus how much the players' speed, the
walking share, the ball's visibility and speed differ between the half-second after and before.
Gradient boosting. Cached as cache/dchange_<variant>.joblib.

Scored against the submission (team from the 9 DFL models, dead ball from the 3 DFL + PFF models,
28 / 23 / 32), with dead-ball cost max(4, dead - bonus * p_boundary); paired on the Metrica
validation set, plus the official window (scripts/pff_train.official_paired).
"""

import os
import re
import sys
import warnings

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
from possession import data, dfl, features  # noqa: E402
import possession_metrics as pm  # noqa: E402
import change_detector as cd  # noqa: E402
from dfl_pretrain import DROP, dfl_features, our_missing_mask  # noqa: E402
from dfl_change import inputs  # noqa: E402
import validate_clean as vc  # noqa: E402
from metrica_lomo import CHOSEN  # noqa: E402
from pff_train import official_paired, pff_ids  # noqa: E402

SUBMISSION = dict(CHOSEN, team_models="plain", dead=24.0)
SHIFTED = ("speed_all_r12", "walk_all_r12", "ball_seen_r12", "ball_speed_r12")


def boundary_targets(lab):
    n = len(lab)
    live, dead = np.isin(lab, (0, 1)), lab == 2
    edge = np.flatnonzero((live[1:] & dead[:-1]) | (dead[1:] & live[:-1])) + 1
    near = np.zeros(n, bool)
    far_ok = np.ones(n, bool)
    for i in edge:
        near[max(0, i - cd.NEAR):i + cd.NEAR + 1] = True
        far_ok[max(0, i - cd.FAR):i + cd.FAR + 1] = False
    graded = lab < 3
    return graded & near, graded & far_ok


def dead_inputs(df, cols):
    extra = [np.abs(cd.window_shift(df[c].to_numpy(), cd.HALF)) for c in SHIFTED]
    return np.hstack([inputs(df, cols), np.column_stack(extra).astype(np.float32)])


def detector(variant, miss, cols):
    path = os.path.join(data.CACHE, "dchange_{}.joblib".format(variant))
    if os.path.exists(path):
        return joblib.load(path)
    ids = {"dfl": list(dfl.MATCHES), "dflpff": list(dfl.MATCHES) + pff_ids(),
           "pff": pff_ids()}[variant]
    Xs, ys = [], []
    for mid in ids:
        is_pff = mid.startswith("P")
        for seed in ((0,) if is_pff else (0, 1)):
            d = dfl_features(mid, seed, miss)
            pos, neg = boundary_targets(d.label.to_numpy())
            keep = np.flatnonzero(pos | neg)
            keep = keep[(keep % (4 if is_pff else 2) == 0) | pos[keep]]   # thin negatives only
            Xs.append(dead_inputs(d, cols)[keep])
            ys.append(pos[keep].astype(int))
    det = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=300,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         class_weight="balanced", early_stopping=False,
                                         random_state=0).fit(np.vstack(Xs), np.concatenate(ys))
    joblib.dump(det, path)
    print("  fitted {} ({} matches)".format(os.path.basename(path), len(ids)), flush=True)
    return det


def submission_samples(team_extra=()):
    team9 = vc.load_models(vc.PLAIN + vc.STATE)
    dead = [joblib.load(os.path.join(data.CACHE, "pff_dflpff_st_s{}.joblib".format(s)))
            for s in (0, 2, 4)]
    return vc.Samples(plain=team9 + list(team_extra), state=dead)


def set_pdc(S, det, miss, cols, ours):
    S.pdc = {"Metrica": [], "match": []}
    for g in ("M1", "M2", "M3"):
        fr = dfl_features(g, 0, miss)
        p = det.predict_proba(dead_inputs(fr, cols))[:, 1]
        S.pdc["Metrica"] += [p[s:e] for s, e in pm.valid_runs(fr.label.to_numpy())]
    p = det.predict_proba(dead_inputs(ours, cols))[:, 1]
    S.pdc["match"] = [p[a:b] for a, b, _, _, _ in S.units["match"]]


def choice(miss, cols, ours):
    """Dead ball from DFL (state noise), DFL + PFF or both averaged, with or without the
    boundary detector, against the best DFL-only setting on the corrected Metrica labels."""
    team9 = vc.load_models(vc.PLAIN + vc.STATE)
    dfl_dead = vc.load_models(vc.STATE)
    pff_dead = [joblib.load(os.path.join(data.CACHE, "pff_dflpff_st_s{}.joblib".format(s)))
                for s in (0, 2, 4)]
    runs = {"DFL": vc.Samples(plain=team9, state=dfl_dead),
            "DFL+PFF": vc.Samples(plain=team9, state=pff_dead),
            "both": vc.Samples(plain=team9, state=dfl_dead + pff_dead)}
    ref = vc.evaluate(runs["DFL"], dict(SUBMISSION, dead=32.0))
    vc.report("reference: dead ball DFL, 28/23/32", ref, ref)
    dets = {v: detector(v, miss, cols) for v in ("dflpff", "pff")}
    for name, S in runs.items():
        for dead in (24.0, 32.0):
            res = vc.evaluate(S, dict(SUBMISSION, dead=dead))
            vc.report("dead ball {}, 28/23/{:.0f}".format(name, dead), ref, res)
            print("    official: " + official_paired(ref, res), flush=True)
        for v, det in dets.items():
            set_pdc(S, det, miss, cols, ours)
            for dead, bonus in ((32.0, 16.0), (32.0, 28.0), (48.0, 44.0)):
                res = vc.evaluate(S, dict(SUBMISSION, dead=dead, dead_bonus=bonus))
                vc.report("dead ball {}, {} detector, dead {:.0f} bonus {:.0f}".format(
                    name, v, dead, bonus), ref, res)
                print("    official: " + official_paired(ref, res), flush=True)
            del S.pdc


def main():
    warnings.filterwarnings("ignore")
    ours, _ = features.build()
    cols = [c for c in ours.columns if not re.search(DROP, c)]
    miss = our_missing_mask()
    if sys.argv[1:] == ["choice"]:
        return choice(miss, cols, ours)
    if sys.argv[1:] == ["proba"]:
        # the detector trained on PFF alone, on our match: cache/pdead_change_pff.npy
        p = detector("pff", miss, cols).predict_proba(dead_inputs(ours, cols))[:, 1]
        np.save(os.path.join(data.CACHE, "pdead_change_pff.npy"), p.astype(np.float32))
        print("wrote cache/pdead_change_pff.npy")
        return
    S = submission_samples()
    ref = vc.evaluate(S, SUBMISSION)
    vc.report("submission", ref, ref)
    if sys.argv[1:] == ["team"]:
        extra = {k: [joblib.load(os.path.join(data.CACHE, "pff_dflpff_{}.joblib".format(k)))]
                 for k in ("plain_s0", "plain_s2", "st_s0", "st_s2", "st_s4")}
        for name, models in (("+ 2 DFL + PFF plain", extra["plain_s0"] + extra["plain_s2"]),
                             ("+ 5 DFL + PFF", sum(extra.values(), []))):
            res = vc.evaluate(submission_samples(models), SUBMISSION)
            vc.report("team: 9 DFL " + name, ref, res)
            print("    official: " + official_paired(ref, res), flush=True)
        return
    for variant in ("dfl", "dflpff", "pff"):
        set_pdc(S, detector(variant, miss, cols), miss, cols, ours)
        for dead, bonus in ((32.0, 16.0), (32.0, 28.0), (48.0, 32.0), (48.0, 44.0),
                            (64.0, 48.0), (64.0, 60.0)):
            res = vc.evaluate(S, dict(SUBMISSION, dead=dead, dead_bonus=bonus))
            vc.report("{} detector, dead {:.0f} bonus {:.0f}".format(variant, dead, bonus), ref,
                      res)
            print("    official: " + official_paired(ref, res), flush=True)


if __name__ == "__main__":
    main()
