"""Learn "who has the ball" from the 7 DFL open matches, apply it to ours.

    .venv/bin/python scripts/dfl_pretrain.py [--copies 2]

1. Each DFL match (possession/dfl.py) is made to look like our tracking (dfl.add_noise: camera
   projection of airborne balls, jitter, the ball missing where ours is missing in a random
   stretch of our match, player fragments), `--copies` times with different seeds, and turned into
   our features (possession/features.build). No Smart Tagging there, so tag signals are left out.
2. A gradient-boosting model learns DFL's per-frame label (home / away / dead). DFL counts every
   touch as a change, so this is a model of "who touched the ball last", not of a possession
   convention.
3. Leave-one-match-out on DFL says how well that is learnable from noisy tracking; then the model
   trained on all 7 is applied to our match: cache/dfl_proba.npy (N_FRAMES, 3), and scored on the
   official window: frame team accuracy, and team choice per true possession.
"""

import argparse
import subprocess
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
from possession import data, dfl, features  # noqa: E402
import possession_metrics as pm  # noqa: E402

DROP = r"^tag_|^setpiece|(^|vis_)touch_|^ev_|^own_rule|^vis_|^aud_|^half$"
STEP = 2            # every 2nd frame for training (neighbours are near duplicates)


def our_missing_mask():
    ball = data.load_tracking()["ball"]
    graded = np.zeros(data.N_FRAMES, bool)
    for _, first, last in data.halves():
        graded[first:last + 1] = True
    return np.isnan(ball[graded, 0])


def state_pools():
    """Our ball's missing mask over live frames and over dead-ball frames. The game state comes
    from the plain DFL model (seed 0, cache/dfl_proba.npy, trained first if missing) decoded with
    the fixed costs 64/128: no labels and no earlier submission involved."""
    from possession import decode
    path = os.path.join(data.CACHE, "dfl_proba.npy")
    if not os.path.exists(path):
        subprocess.run([sys.executable, os.path.abspath(__file__), "--no-lomo"], check=True)
    proba = np.load(path).astype(np.float64)
    state = np.full(data.N_FRAMES, 3, np.int8)
    for _, first, last in data.halves():
        state[first:last + 1] = decode.viterbi(proba[first:last + 1], 64.0, 128.0)
    miss = np.isnan(data.load_tracking()["ball"][:, 0])
    return miss[np.isin(state, (0, 1))], miss[state == 2]


def mirrored(m):
    """The same match with home and away swapped (team ids and labels)."""
    out = dict(m)
    out["team"] = (1 - m["team"]).astype(np.int8)
    lab = m["labels"].copy()
    lab[m["labels"] == 0] = 1
    lab[m["labels"] == 1] = 0
    out["labels"] = lab
    return out


CALIBRATED = dict(ball_white=0.05, player_drop=0.02, player_smooth=True, air_miss=0.8,
                  ball_sigma=0.3)        # scripts/calibrate_noise.py, configuration G


def dfl_features(match_id, seed, miss, pools=None, mirror=False, calibrated=False):
    """Features of a noisy copy of a DFL match (or, for ids "M1", "M2", of a Metrica open match;
    for ids "P<game>", of a PFF World Cup match).
    `miss`: our missing mask (random placement), or `pools` (live, dead) for state-dependent
    missingness. Metrica's and PFF's own balls already go missing like ours: no extra gaps there.
    `mirror`: home and away swapped. Cached per variant."""
    metrica, pff_match = match_id.startswith("M"), match_id.startswith("P")
    path = os.path.join(data.CACHE, "{}_feat_{}_{}{}{}{}.pkl".format(
        "met" if metrica else "pff" if pff_match else "dfl", match_id, seed,
        "_st" if pools is not None else "", "_mir" if mirror else "", "_cal" if calibrated else ""))
    if os.path.exists(path):
        return metrica_rule(pd.read_pickle(path), match_id)
    m, noisy = noisy_copy(match_id, seed, miss, pools, mirror, calibrated)
    df, _ = features.build(match=noisy)
    df["label"] = m["labels"]
    df.to_pickle(path)
    return metrica_rule(df, match_id)


def metrica_rule(df, match_id):
    """With METRICA_LABELS=lite|strict (PFF_LABELS for the PFF matches), a Metrica (PFF) match's
    labels follow the challenge's "controlled possession" rule
    (possession/metrica_open.CONTROL_RULES) instead of every controlled action."""
    if match_id.startswith("M"):
        rule = os.environ.get("METRICA_LABELS", "events")
        if rule != "events":
            from possession import metrica_open
            df = df.copy()
            df["label"] = metrica_open.labels(match_id, rule)
    elif match_id.startswith("P"):
        rule = os.environ.get("PFF_LABELS", "events")
        if rule != "events":
            from possession import pff
            df = df.copy()
            df["label"] = pff.labels(match_id[1:], rule)
    return df


def noisy_copy(match_id, seed, miss, pools=None, mirror=False, calibrated=False):
    """(clean match, noisy copy) as dfl_features builds them; deterministic for a seed."""
    if match_id.startswith(("M", "P")):
        if match_id.startswith("M"):
            from possession import metrica_open
            m = (metrica_open.load_game3() if match_id == "M3"
                 else metrica_open.load(int(match_id[1:])))
        else:
            from possession import pff
            m = pff.load(match_id[1:])
        noisy = dfl.add_noise(m, np.zeros(len(m["ball"]), bool), seed=seed,
                              **(CALIBRATED if calibrated else {}))
    else:
        m = dfl.load(match_id)
        noisy = dfl.add_noise(m, miss, seed=seed, state_pools=pools,
                              **(CALIBRATED if calibrated else {}))
    if mirror:
        m, noisy = mirrored(m), mirrored(noisy)
    return m, noisy


def pass_features(match_id, keep=1.0):
    """possession/features_pass.py for a match (seed-independent). DFL / Metrica / PFF ids: from
    the CLEAN tracking, because pass detection on our real tracking is as reliable as on clean data
    (completed passes inside the passer's possession: official window 97%, clean Metrica 96%,
    Metrica with our simulated noise 75%). "ours": our tracking. `keep` < 1: detected flights
    randomly kept (our tracking detects ~2/3 of the passes clean data does). Cached as
    cache/<prefix>_pfeat_<id>[_k<keep>].pkl."""
    from possession import features_pass
    prefix = {"M": "met", "P": "pff"}.get(match_id[0], "dfl") if match_id != "ours" else "ours"
    path = os.path.join(data.CACHE, "{}_pfeat_{}{}.pkl".format(
        prefix, match_id, "" if keep >= 1.0 else "_k{:g}".format(keep)))
    if os.path.exists(path):
        return pd.read_pickle(path)
    if match_id == "ours":
        m = data.load_tracking()
    else:
        m, _ = noisy_copy(match_id, 0, np.zeros(1, bool) if match_id[0] in "MP" else
                          our_missing_mask())
    df = features_pass.build_pass(m, keep=keep if match_id != "ours" else 1.0)
    df.to_pickle(path)
    return df


def extra_features(match_id, seed, miss, pools=None, calibrated=False):
    """possession/features_extra.py on the same noisy copy as dfl_features(match_id, seed, ...);
    match_id "ours": our match as tracked. Cached as
    cache/<prefix>_xfeat_<id>_<seed>[_st][_cal].pkl."""
    from possession import features_extra
    if match_id == "ours":
        path = os.path.join(data.CACHE, "features_extra_clean.pkl")
        if os.path.exists(path):
            return pd.read_pickle(path)
        t = data.load_tracking()
        sign = features.home_attack_sign(t["players"], t["team"], None)
        half = data.half_index()
        hsign = np.ones(len(t["players"]), np.float32)
        for h, sg in sign.items():
            hsign[half == h] = sg
        df = features_extra.build_extra(t, hsign=hsign)
        df.to_pickle(path)
        return df
    prefix = {"M": "met", "P": "pff"}.get(match_id[0], "dfl")
    path = os.path.join(data.CACHE, "{}_xfeat_{}_{}{}{}.pkl".format(
        prefix, match_id, seed, "_st" if pools is not None else "", "_cal" if calibrated else ""))
    if os.path.exists(path):
        return pd.read_pickle(path)
    _, noisy = noisy_copy(match_id, seed, miss, pools, calibrated=calibrated)
    df = features_extra.build_extra(noisy)
    df.to_pickle(path)
    return df


def make_model():
    return HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.1, max_iter=300, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def team_by_true_possession(gt, proba):
    rows = []
    for s, e, k in pm.segments(gt, pm.valid_runs(gt)):
        if k == pm.D:
            continue
        p = proba[s:e, :2].mean(0)
        rows.append(((e - s) / data.FPS, int(np.argmax(p) == k)))
    d = pd.DataFrame(rows, columns=["dur", "ok"])
    return d.groupby(pd.cut(d.dur, [0, 1, 4, 1e9], labels=["<1 s", "1-4 s", ">4 s"]),
                     observed=False).ok.agg(["size", "mean"]).round(2)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--copies", type=int, default=2)
    p.add_argument("--first-seed", type=int, default=0, help="noise seeds first..first+copies-1")
    p.add_argument("--no-lomo", action="store_true",
                   help="skip leave-one-match-out on DFL (and reuse the saved model)")
    p.add_argument("--out", default="dfl_proba.npy")
    p.add_argument("--state-noise", action="store_true",
                   help="ball missing as in our match in the same game state (live / dead), "
                        "the state taken from the plain DFL model (no labels)")
    p.add_argument("--metrica", action="store_true", help="add the 3 Metrica open matches")
    p.add_argument("--metrica-only", action="store_true", help="train on the Metrica matches only")
    p.add_argument("--mirror", action="store_true", help="add home/away-swapped copies")
    p.add_argument("--calibrated", action="store_true",
                   help="noise calibrated on our tracking's statistics (CALIBRATED knobs)")
    p.add_argument("--touch", action="store_true",
                   help="keep the touch features (sudden ball velocity changes next to a player)")
    p.add_argument("--target", default="touch",
                   help="DFL label to learn: touch (DFL's per-frame possession), events "
                        "(controlled actions only), sandwich<s> (opponent spells < s removed)")
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    miss = our_missing_mask()
    pools = state_pools() if args.state_noise else None
    print("DFL features ...", flush=True)
    seeds = range(args.first_seed, args.first_seed + args.copies)
    ids = [] if args.metrica_only else list(dfl.MATCHES)
    if args.metrica or args.metrica_only:
        ids += ["M1", "M2", "M3"]
    mirrors = (False, True) if args.mirror else (False,)
    sets = {mid: [dfl_features(mid, seed, miss, pools if not mid.startswith("M") else None, mir,
                               calibrated=args.calibrated)
                  for seed in seeds for mir in mirrors] for mid in ids}
    ours, _ = features.build()
    drop = DROP.replace(r"(^|vis_)touch_|", "") if args.touch else DROP
    cols = [c for c in ours.columns if not re.search(drop, c)]
    if args.metrica_only:
        args.no_lomo = True

    if args.target != "touch":
        for mid in [m for m in ids if not m.startswith("M")]:
            lab = dfl.labels_variant(mid, dfl.load(mid), args.target)
            for d in sets[mid]:
                d["label"] = lab

    def stack(ids):
        dfs = [d.iloc[::STEP] for mid in ids if mid in sets for d in sets[mid]]
        df = pd.concat(dfs)
        df = df[df.label < 3]
        return df[cols].to_numpy(np.float32), df.label.to_numpy()

    if not args.no_lomo:
        accs = []
        for mid in dfl.MATCHES:
            X, y = stack([m for m in dfl.MATCHES if m != mid])
            mdl = make_model().fit(X, y)
            d = sets[mid][0]
            d = d[d.label < 3]
            pr = mdl.predict_proba(d[cols].to_numpy(np.float32))
            live = d.label.to_numpy() < 2
            team_acc = np.mean(np.argmax(pr[live, :2], 1) == d.label.to_numpy()[live])
            dead_acc = np.mean((pr[:, 2] > 0.5) == (d.label.to_numpy() == 2))
            accs.append((team_acc, dead_acc))
            print("  held-out {}: team acc {:.3f}, dead/alive acc {:.3f}".format(
                mid, team_acc, dead_acc), flush=True)
        print("DFL leave-one-match-out: team {:.3f}, dead/alive {:.3f}".format(
            *np.mean(accs, axis=0)))

    import joblib
    model_path = os.path.join(data.CACHE, "dfl_model{}{}{}{}.joblib".format(
        "" if args.target == "touch" else "_" + args.target, "_st" if args.state_noise else "",
        "" if args.first_seed == 0 else "_s{}".format(args.first_seed),
        "".join(t for t, on in (("_met", args.metrica), ("_metonly", args.metrica_only),
                                ("_mir", args.mirror), ("_touch", args.touch),
                                ("_cal", args.calibrated)) if on)))
    if os.path.exists(model_path) and args.no_lomo:
        mdl = joblib.load(model_path)
    else:
        X, y = stack(ids)
        mdl = make_model().fit(X, y)
        joblib.dump(mdl, model_path)
    proba = mdl.predict_proba(ours[cols].to_numpy(np.float32))
    np.save(os.path.join(data.CACHE, args.out), proba.astype(np.float32))

    from error_anatomy import truth
    gt = truth()
    live = np.isin(gt, [pm.H, pm.A])
    print("\nOn the official window (never trained on), DFL model alone:")
    print("  frame team accuracy on live frames: {:.3f}".format(
        np.mean(np.argmax(proba[live, :2], 1) == gt[live])))
    print("  team choice given the true possessions:")
    print(team_by_true_possession(gt, proba).to_string())


if __name__ == "__main__":
    main()
