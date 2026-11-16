"""A larger validation set in Metrica's convention: the 3 open Metrica matches, made noisy like our
tracking, predicted by DFL-trained models, scored against Metrica's own (event-based) possession.

    .venv/bin/python scripts/metrica_val.py

~1,100 possessions in the provider's convention (RECOVERY marks a gain). Never used for training
here: the DFL models do not see Metrica. Variants of the submission's decoding / ensemble are
scored on it.
"""

import os
import re
import sys
import warnings

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
from possession import data, decode, features, metrica_open  # noqa: E402
import possession_metrics as pm  # noqa: E402
from dfl_pretrain import DROP, dfl_features, our_missing_mask  # noqa: E402
from dfl_change import inputs  # noqa: E402

PLAIN = ["dfl_model.joblib"] + ["dfl_model_s{}.joblib".format(s) for s in (2, 4, 6, 8, 10)]
STATE = ["dfl_model_st.joblib", "dfl_model_st_s2.joblib", "dfl_model_st_s4.joblib"]
GAMES = ("M1", "M2", "M3")


def bag(X, plain, state):
    pp = [joblib.load(os.path.join(data.CACHE, m)).predict_proba(X) for m in plain]
    ps = [joblib.load(os.path.join(data.CACHE, m)).predict_proba(X) for m in state]
    dead = np.mean([p[:, 2] for p in ps], axis=0)
    home = np.mean([p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-9) for p in pp + ps], axis=0)
    return np.column_stack([(1 - dead) * home, (1 - dead) * (1 - home), dead])


def spell(lab, proba, pc, team_cost=64.0, bonus=32.0, dead_cost=128.0):
    runs = pm.valid_runs(lab)
    pred = np.full(len(lab), pm.X, np.int8)
    for s, e in runs:
        tc = np.maximum(4.0, team_cost - bonus * pc[s:e])
        dd = np.full(e - s, dead_cost)
        pred[s:e] = decode.viterbi_tv(proba[s:e], tc, dd, dd)
    return pm.spell_scores(lab, pred, runs)


def main():
    warnings.filterwarnings("ignore")
    ours, _ = features.build()
    cols = [c for c in ours.columns if not re.search(DROP, c)]
    det, dcols = joblib.load(os.path.join(data.CACHE, "dfl_change_touch.joblib"))
    miss = our_missing_mask()
    games = {}
    for g in GAMES:
        d = dfl_features(g, 0, miss)
        lab = d.label.to_numpy().astype(np.int8)
        X = d[cols].to_numpy(np.float32)
        games[g] = (lab, X, det.predict_proba(inputs(d, dcols))[:, 1])
    n_poss = sum(len(pm.possession_spells(l, pm.valid_runs(l))) for l, _, _ in games.values())
    print("Metrica validation: {} games, {} possessions".format(len(games), n_poss))
    variants = {
        "submission (team 9, dead 3 st), 64/32/128": dict(plain=PLAIN, state=STATE),
        "  no change signal (bonus 0)": dict(plain=PLAIN, state=STATE, bonus=0.0),
        "  team cost 32": dict(plain=PLAIN, state=STATE, team_cost=32.0),
        "  team cost 96": dict(plain=PLAIN, state=STATE, team_cost=96.0),
        "  dead cost 64": dict(plain=PLAIN, state=STATE, dead_cost=64.0),
        "  dead cost 256": dict(plain=PLAIN, state=STATE, dead_cost=256.0),
        "  plain models only (6)": dict(plain=PLAIN, state=[], dead_from_plain=True),
    }
    for name, v in variants.items():
        scores, tp, npred, ngt = [], 0, 0, 0
        for g, (lab, X, pc) in games.items():
            if v.get("dead_from_plain"):
                proba = bag(X, v["plain"], v["plain"])
            else:
                proba = bag(X, v["plain"], v["state"])
            r = spell(lab, proba, pc, v.get("team_cost", 64.0), v.get("bonus", 32.0),
                      v.get("dead_cost", 128.0))
            scores.append(r["mean_f1"])
            npred += r["n_pred"]
            ngt += r["n_gt"]
        print("{:<46s} spellF1 {:.3f}  (per game {})  possessions {}/{}".format(
            name, np.mean(scores), " ".join("{:.3f}".format(s) for s in scores), npred, ngt),
            flush=True)


if __name__ == "__main__":
    main()
