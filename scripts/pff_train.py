"""Does the PFF World Cup 2022 data help? 64 more matches, labelled in an event-based convention
like the provider's (possession/pff.py).

    .venv/bin/python scripts/pff_train.py features     # build the noisy PFF copies (seeds 0, 1)
    .venv/bin/python scripts/pff_train.py frame        # frame models
    .venv/bin/python scripts/pff_train.py change       # change detector
    .venv/bin/python scripts/pff_train.py mix          # team / dead ball from which models; costs
    .venv/bin/python scripts/pff_train.py dead         # dead ball from PFF in the full bag
    .venv/bin/python scripts/pff_train.py proba SEED   # the submission's dead-ball member SEED

frame: the 2 + 2 models of scripts/metrica_lomo.py (plain and state-noise, noise seeds 0 and 2) are
refitted on DFL + PFF and on PFF alone, and compared, paired, with the submission's models of the
same noise draws (DFL alone). PFF frames are thinned to one in PFF_STEP and take noise draw
`seed` only (DFL: every 2nd frame, draws seed and seed + 1). PFF's ball already goes missing by
game state, so the state-noise models differ from the plain ones on DFL frames only.
change: the change detector refitted on DFL + PFF changes and on PFF alone, against the DFL one,
with the submission's 9 frame models.
Both scored on the Metrica validation set (never trained on) and the official window, decoded
like the submission. Models cached as cache/pff_<variant>_<kind>_s<seed>.joblib.
"""

import os
import re
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
from possession import data, dfl, features, pff  # noqa: E402
from dfl_pretrain import (DROP, STEP, dfl_features, make_model, our_missing_mask,  # noqa: E402
                          state_pools)
from dfl_change import inputs, targets  # noqa: E402
import validate_clean as vc  # noqa: E402
from metrica_lomo import CHOSEN, REF, SEEDS  # noqa: E402

PFF_STEP = 4


def pff_ids():
    return ["P" + g for g in pff.games()]


def rows(ids, kind, seed, miss, pools, cols):
    dfs = []
    for mid in ids:
        is_pff = mid.startswith("P")
        for s in ((seed,) if is_pff else (seed, seed + 1)):
            st = pools if (kind == "st" and not is_pff) else None
            d = dfl_features(mid, s, miss, st).iloc[::PFF_STEP if is_pff else STEP]
            dfs.append(d[d.label < 3])
    df = pd.concat(dfs)
    return df[cols].to_numpy(np.float32), df.label.to_numpy()


def frame_model(variant, kind, seed, miss, pools, cols):
    path = os.path.join(data.CACHE, "pff_{}_{}_s{}.joblib".format(variant, kind, seed))
    if os.path.exists(path):
        return joblib.load(path)
    ids = pff_ids() + (list(dfl.MATCHES) if variant == "dflpff" else [])
    X, y = rows(ids, kind, seed, miss, pools, cols)
    mdl = make_model().fit(X, y)
    joblib.dump(mdl, path)
    print("  fitted {} ({:,} rows, {} matches)".format(os.path.basename(path), len(y), len(ids)),
          flush=True)
    return mdl


def change_model(variant, miss, cols):
    path = os.path.join(data.CACHE, "pff_{}_change.joblib".format(variant))
    if os.path.exists(path):
        return joblib.load(path), cols
    Xs, ys = [], []
    ids = pff_ids() + (list(dfl.MATCHES) if variant == "dflpff" else [])
    for mid in ids:
        is_pff = mid.startswith("P")
        lab = (dfl_features(mid, 0, miss).label.to_numpy() if is_pff
               else dfl.labels_variant(mid, dfl.load(mid), "touch"))
        pos, neg = targets(lab)
        keep = np.flatnonzero(pos | neg)
        keep = keep[(keep % (4 if is_pff else 2) == 0) | pos[keep]]   # thin negatives only
        for seed in ((0,) if is_pff else (0, 1)):
            Xs.append(inputs(dfl_features(mid, seed, miss), cols)[keep])
            ys.append(pos[keep].astype(int))
    det = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=300,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         class_weight="balanced", early_stopping=False,
                                         random_state=0).fit(np.vstack(Xs), np.concatenate(ys))
    joblib.dump(det, path)
    print("  fitted {} ({} matches)".format(os.path.basename(path), len(ids)), flush=True)
    return det, cols


def official_paired(ref, res, draws=2000):
    """spellF1 difference over the official window's possessions and macroF1 difference over
    its 30 s blocks, 90% intervals."""
    from compare import f1
    from sklearn.metrics import f1_score
    (fa, na), (fb, nb) = ref[9], res[9]
    n = fa.shape[1]
    rng = np.random.RandomState(0)
    d = [f1(fb, nb, k) - f1(fa, na, k) for k in (rng.randint(0, n, n) for _ in range(draws))]
    gt = pff_official()
    idx = np.flatnonzero(gt != 3)
    blocks = np.array_split(idx, max(1, len(idx) // (30 * data.FPS)))
    a, b = ref[10], res[10]

    def macro(sel):
        f = np.concatenate([blocks[i] for i in sel])
        return f1_score(gt[f], b[f], labels=[0, 1, 2], average="macro") - f1_score(
            gt[f], a[f], labels=[0, 1, 2], average="macro")
    m = [macro(rng.randint(0, len(blocks), len(blocks))) for _ in range(500)]
    return "spellF1 {:+.3f} [{:+.3f}, {:+.3f}] | macroF1 {:+.3f} [{:+.3f}, {:+.3f}]".format(
        f1(fb, nb, np.arange(n)) - f1(fa, na, np.arange(n)), *np.percentile(d, [5, 95]),
        macro(range(len(blocks))), *np.percentile(m, [5, 95]))


def pff_official():
    import possession_metrics as pm
    return pm.label_codes(data.load_public_gt())


def main():
    warnings.filterwarnings("ignore")
    what = sys.argv[1] if len(sys.argv) > 1 else "frame"
    ours, _ = features.build()
    cols = [c for c in ours.columns if not re.search(DROP, c)]
    miss = our_missing_mask()
    if what == "proba":
        # state-noise model on DFL + PFF, noise seeds SEED and SEED + 1 (DFL), SEED (PFF): its
        # probabilities on our match, cache/pff_proba_st_s<SEED>.npy (scripts/make_submission.py)
        seed = int(sys.argv[2])
        mdl = frame_model("dflpff", "st", seed, miss, state_pools(), cols)
        out = "pff_proba_st_s{}.npy".format(seed)
        np.save(os.path.join(data.CACHE, out),
                mdl.predict_proba(ours[cols].to_numpy(np.float32)).astype(np.float32))
        print("wrote cache/" + out)
        return
    if what == "features":
        for i, mid in enumerate(pff_ids()):
            for seed in (0, 1):
                dfl_features(mid, seed, miss)
            if i % 8 == 0:
                print("  {} / {} matches".format(i + 1, len(pff_ids())), flush=True)
        lab = np.concatenate([dfl_features(m, 0, miss).label.to_numpy() for m in pff_ids()])
        print("PFF: {} matches, {:.0f} h of frames; H {:.0%} A {:.0%} D {:.0%} X {:.0%}".format(
            len(pff_ids()), len(lab) / data.FPS / 3600, *(np.mean(lab == k) for k in range(4))))
        return
    if what == "frame":
        pools = state_pools()
        ref_models = {k: joblib.load(os.path.join(data.CACHE, v)) for k, v in REF.items()}
        S = vc.Samples(plain=[ref_models[("plain", s)] for s in SEEDS],
                       state=[ref_models[("st", s)] for s in SEEDS])
        ref = vc.evaluate(S, CHOSEN)
        vc.report("reference: DFL only (2 + 2 models)", ref, ref)
        for variant in ("dflpff", "pffonly"):
            m = {(k, s): frame_model(variant, k, s, miss, pools, cols)
                 for k in ("plain", "st") for s in SEEDS}
            V = vc.Samples(plain=[m[("plain", s)] for s in SEEDS],
                           state=[m[("st", s)] for s in SEEDS])
            vc.report({"dflpff": "DFL + PFF", "pffonly": "PFF only"}[variant], ref,
                      vc.evaluate(V, CHOSEN))
    elif what == "mix":
        # which part helps: team from the plain models, dead ball from the state-noise ones, each
        # from DFL alone or from DFL + PFF; then the decoder's costs for the best mixture
        ref_models = {k: joblib.load(os.path.join(data.CACHE, v)) for k, v in REF.items()}
        new = {(k, s): frame_model("dflpff", k, s, miss, None, cols)
               for k in ("plain", "st") for s in SEEDS}
        plain_only = dict(CHOSEN, team_models="plain")
        runs = {}
        for team in ("DFL", "DFL+PFF"):
            for dead in ("DFL", "DFL+PFF"):
                src = {"DFL": ref_models, "DFL+PFF": new}
                runs[(team, dead)] = vc.Samples(
                    plain=[src[team][("plain", s)] for s in SEEDS],
                    state=[src[dead][("st", s)] for s in SEEDS])
        ref = vc.evaluate(runs[("DFL", "DFL")], plain_only)
        for (team, dead), S in runs.items():
            vc.report("team {}, dead ball {}".format(team, dead), ref,
                      vc.evaluate(S, plain_only))
        for key in (("DFL+PFF", "DFL"), ("DFL+PFF", "DFL+PFF"), ("DFL", "DFL+PFF")):
            for tc in (20.0, 28.0, 40.0, 56.0):
                for dead in (32.0, 64.0, 128.0):
                    cfg = dict(plain_only, team=tc, bonus=round(tc * 0.82), dead=dead)
                    vc.report("  team {}, dead {}: costs {:.0f}/{:.0f}/{:.0f}".format(
                        *key, tc, cfg["bonus"], dead), ref, vc.evaluate(runs[key], cfg))
    elif what == "dead":
        # the submission's 9 DFL models for the team; dead ball from 3 state-noise models trained
        # on DFL + PFF, 3 on DFL alone (state noise or random gaps), or 2 on PFF alone; costs grid.
        # Paired on Metrica against the submission, plus the official window's possessions and
        # macroF1 by 30 s blocks
        pools = state_pools()
        team9 = vc.load_models(vc.PLAIN + vc.STATE)
        dead = {"DFL+PFF": [frame_model("dflpff", "st", s, miss, pools, cols) for s in (0, 2, 4)],
                "DFL state": vc.load_models(vc.STATE),
                "DFL plain": vc.load_models(vc.PLAIN[:3]),
                "PFF only": [frame_model("pffonly", "st", s, miss, pools, cols) for s in SEEDS]}
        base = dict(CHOSEN, team_models="plain", dead=32.0)
        runs = {k: vc.Samples(plain=team9, state=v) for k, v in dead.items()}
        ref = vc.evaluate(runs["DFL+PFF"], base)
        vc.report("submission (dead ball DFL+PFF, 28/23/32)", ref, ref)
        for name, S in runs.items():
            for tc in (20.0, 28.0, 36.0, 48.0):
                for dc in (24.0, 32.0, 48.0, 64.0, 96.0, 128.0):
                    cfg = dict(base, team=tc, bonus=round(tc * 0.82), dead=dc)
                    res = vc.evaluate(S, cfg)
                    vc.report("{} {:.0f}/{:.0f}/{:.0f}".format(name, tc, cfg["bonus"], dc), ref,
                              res)
                    print("    official: " + official_paired(ref, res), flush=True)
    elif what == "change":
        S = vc.Samples()
        ref = vc.evaluate(S, CHOSEN)
        vc.report("reference: DFL change detector", ref, ref)
        for variant in ("dflpff", "pffonly"):
            V = vc.Samples(change=change_model(variant, miss, cols))
            vc.report("change detector on {}".format(
                {"dflpff": "DFL + PFF", "pffonly": "PFF only"}[variant]), ref,
                vc.evaluate(V, CHOSEN))


if __name__ == "__main__":
    main()
