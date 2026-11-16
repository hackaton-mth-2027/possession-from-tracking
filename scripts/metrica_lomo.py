"""Does training on Metrica's own matches help? Leave one Metrica match out.

    .venv/bin/python scripts/metrica_lomo.py            # frame models
    .venv/bin/python scripts/metrica_lomo.py change     # change detector

The Metrica open matches are the only data in the provider's convention, and also our validation
set, so each is scored by models trained without it: DFL (7 matches) + the two other Metrica
matches, Metrica frames weighted W. Reference: the same noise draws on DFL alone (the submission's
dfl_model / dfl_model_s2 for the team, dfl_model_st / dfl_model_st_s2 for the dead ball), so only
the added Metrica data differs. Decoded like the submission (scripts/validate_clean.py, chosen
costs), change probabilities from the DFL detector. Paired over the 1,117 possessions.
Fitted models are cached as cache/lomo_<held-out>_<kind>_s<seed>_w<W>.joblib.

`change`: the same for the change detector (scripts/dfl_change.py): DFL's changes + the two other
Metrica matches' changes (weight W), against the DFL detector, with the submission's 9 frame
models.
"""

import os
import re
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
from possession import data, dfl, features  # noqa: E402
import possession_metrics as pm  # noqa: E402
from dfl_pretrain import (DROP, STEP, dfl_features, make_model, our_missing_mask,  # noqa: E402
                          state_pools)
from dfl_change import inputs, targets  # noqa: E402
from compare import flags_and_preds  # noqa: E402
import validate_clean as vc  # noqa: E402

GAMES = ("M1", "M2", "M3")
SEEDS = (0, 2)                     # each model sees two noise copies: seed, seed + 1
WEIGHTS = (1.0, 4.0)
CHOSEN = dict(vc.CURRENT, step=1, team=28.0, bonus=23.0, dead=64.0)
REF = {("plain", 0): "dfl_model.joblib", ("plain", 2): "dfl_model_s2.joblib",
       ("st", 0): "dfl_model_st.joblib", ("st", 2): "dfl_model_st_s2.joblib"}


def training_rows(ids, kind, seed, miss, pools, cols):
    dfs = []
    for mid in ids:
        for s in (seed, seed + 1):
            st = pools if (kind == "st" and not mid.startswith("M")) else None
            d = dfl_features(mid, s, miss, st).iloc[::STEP]
            dfs.append(d[d.label < 3].assign(is_met=mid.startswith("M")))
    df = pd.concat(dfs)
    return df[cols].to_numpy(np.float32), df.label.to_numpy(), df.is_met.to_numpy()


def lomo_model(held, kind, seed, w, miss, pools, cols):
    path = os.path.join(data.CACHE, "lomo_{}_{}_s{}_w{:g}.joblib".format(held, kind, seed, w))
    if os.path.exists(path):
        return joblib.load(path)
    ids = list(dfl.MATCHES) + [g for g in GAMES if g != held]
    X, y, met = training_rows(ids, kind, seed, miss, pools, cols)
    mdl = make_model().fit(X, y, sample_weight=np.where(met, w, 1.0))
    joblib.dump(mdl, path)
    print("  fitted {} ({:,} rows, {:.0%} Metrica)".format(os.path.basename(path), len(y),
                                                          met.mean()), flush=True)
    return mdl


def change_detector(held, w, miss, cols):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rule = os.environ.get("METRICA_LABELS", "events")
    path = os.path.join(data.CACHE, "lomo_{}_change_w{:g}{}.joblib".format(
        held, w, "" if rule == "events" else "_" + rule))
    if os.path.exists(path):
        return joblib.load(path)
    Xs, ys, ws = [], [], []
    for mid in list(dfl.MATCHES) + [g for g in GAMES if g != held]:
        met = mid.startswith("M")
        lab = (dfl_features(mid, 0, miss).label.to_numpy() if met
               else dfl.labels_variant(mid, dfl.load(mid), "touch"))
        pos, neg = targets(lab)
        keep = np.flatnonzero(pos | neg)
        keep = keep[(keep % 2 == 0) | pos[keep]]          # thin negatives, keep every positive
        for seed in (0, 1):
            Xs.append(inputs(dfl_features(mid, seed, miss), cols)[keep])
            ys.append(pos[keep].astype(int))
            ws.append(np.full(len(keep), w if met else 1.0))
    det = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=300,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         class_weight="balanced", early_stopping=False,
                                         random_state=0)
    det.fit(np.vstack(Xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
    joblib.dump(det, path)
    print("  fitted " + os.path.basename(path), flush=True)
    return det


def change_main():
    S = vc.Samples()
    miss = our_missing_mask()
    ref = vc.evaluate(S, CHOSEN)
    vc.report("reference: DFL change detector", ref, ref)
    for w in WEIGHTS:
        units = []
        for g in GAMES:
            det = change_detector(g, w, miss, S.cols)
            fr = dfl_features(g, 0, miss)
            pc = det.predict_proba(inputs(fr, S.cols))[:, 1]
            lab = fr.label.to_numpy().astype(np.int8)
            units += [pc[s:e] for s, e in pm.valid_runs(lab)]
        old = S.units["Metrica"]
        S.units["Metrica"] = [(lab, pp, ps, pc) for (lab, pp, ps, _), pc in zip(old, units)]
        vc.report("+ Metrica changes, weight {:g}".format(w), ref, vc.evaluate(S, CHOSEN))
        S.units["Metrica"] = old


def main():
    if sys.argv[1:] == ["change"]:
        return change_main()
    warnings.filterwarnings("ignore")
    ours, _ = features.build()
    cols = [c for c in ours.columns if not re.search(DROP, c)]
    miss, pools = our_missing_mask(), state_pools()
    det, dcols = joblib.load(os.path.join(data.CACHE, "dfl_change_touch.joblib"))
    ref_models = {k: joblib.load(os.path.join(data.CACHE, v)) for k, v in REF.items()}

    def score(models_for):
        """models_for(held, kind, seed) -> model; flags over every Metrica possession."""
        fl, npd = [], []
        for g in GAMES:
            fr = dfl_features(g, 0, miss)
            X = fr[cols].to_numpy(np.float32)
            lab = fr.label.to_numpy().astype(np.int8)
            pp = [models_for(g, "plain", s).predict_proba(X) for s in SEEDS]
            ps = [models_for(g, "st", s).predict_proba(X) for s in SEEDS]
            pc = det.predict_proba(inputs(fr, dcols))[:, 1]
            proba = vc.mix(pp, ps, "all", "st")
            for s, e in pm.valid_runs(lab):
                pred = vc.decode_unit(proba[s:e], pc[s:e], CHOSEN)
                f, n = flags_and_preds(lab[s:e], pred, [(0, e - s)])
                fl.append(f)
                npd.append(n)
        return np.hstack(fl), np.concatenate(npd)

    ref = score(lambda g, kind, s: ref_models[(kind, s)])
    print("reference (DFL only, 2 + 2 models): Metrica {:.3f}".format(
        vc.paired(ref, ref)[0]), flush=True)
    for w in WEIGHTS:
        for which in ("plain", "st", "both"):
            def models_for(g, kind, s, which=which, w=w):
                use = which in ("both", kind)
                return (lomo_model(g, kind, s, w, miss, pools, cols) if use
                        else ref_models[(kind, s)])
            sc, diff, (lo, hi) = vc.paired(ref, score(models_for))
            name = {"plain": "plain (team only)", "st": "state-noise (dead + team)",
                    "both": "all"}[which]
            print("+ Metrica (weight {:g}) in the {} models: Metrica {:.3f} {:+.3f} "
                  "[{:+.3f}, {:+.3f}]".format(w, name, sc, diff, lo, hi), flush=True)


if __name__ == "__main__":
    main()
