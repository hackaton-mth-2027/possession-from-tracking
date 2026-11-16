"""New team signals and a metric-shaped loss for the frame models.

    .venv/bin/python scripts/team_features.py
    .venv/bin/python scripts/team_features.py bag    # full bags: boosting + temporal networks
    .venv/bin/python scripts/team_features.py proba  # the submission's 9 boosting team models

Team models (plain noise, draws seed and seed + 1, every 2nd frame of the 7 DFL matches, as
scripts/dfl_pretrain.py) for seeds 0 and 2, against the submission's dfl_model / dfl_model_s2
(same rows, same settings):
  extra     + possession/features_extra.py (interception, carrier, goalkeeper)
  weighted  frames weighted by min(4, max(0.25, 4 s / length of their possession)): spellF1
            counts possessions, not frames, so short possessions should count as much as long
  both      the two together
Team from the 2 models, everything else as the submission (dead ball from DFL + PFF, boundary
detector, 28 / 23 / 24 / 12); paired on the Metrica validation set, plus the official window.
Cached as cache/team_<variant>_s<seed>.joblib.
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
from dfl_pretrain import (DROP, STEP, dfl_features, extra_features, make_model,  # noqa: E402
                          our_missing_mask, state_pools)
import validate_clean as vc  # noqa: E402
from dead_change import detector, set_pdc  # noqa: E402
from pff_train import official_paired  # noqa: E402
from team_retest import FULL, subset  # noqa: E402

SEEDS = (0, 2)
VARIANTS = ("extra", "weighted", "both")


def possession_weights(lab):
    w = np.ones(len(lab))
    for s, e, k in pm.segments(lab.astype(np.int8), pm.valid_runs(lab.astype(np.int8))):
        if k in (0, 1):
            w[s:e] = min(4.0, max(0.25, 4.0 * data.FPS / (e - s)))
    return w


def frame(match_id, seed, miss, extra, pools=None):
    d = dfl_features(match_id, seed, miss, pools)
    if extra:
        d = pd.concat([d, extra_features(match_id, seed, miss, pools)], axis=1)
    return d


def fit(variant, seed, miss, cols, pools=None, metrica=(), met_weight=4.0, target="touch"):
    """`pools`: the state-noise copies (dfl_pretrain --state-noise), cached as ..._st_s<seed>.
    `metrica`: Metrica matches added to the training rows (labels: METRICA_LABELS), their frames
    weighted `met_weight`, cached as ..._met<games>_<labels>. `target`: the DFL label learned
    (possession/dfl.labels_variant: touch, or lite / strict, the controlled-possession rule)."""
    rule = os.environ.get("METRICA_LABELS", "events")
    path = os.path.join(data.CACHE, "team_{}{}_s{}{}{}.joblib".format(
        variant, "_st" if pools is not None else "", seed,
        "_met{}_{}".format("".join(g[1] for g in metrica), rule) if metrica else "",
        "" if target == "touch" else "_t" + target))
    if os.path.exists(path):
        return joblib.load(path)
    Xs, ys, ws = [], [], []
    for g in metrica:
        for s in (seed, seed + 1):
            d = frame(g, s, miss, variant in ("extra", "both"))
            d = d.iloc[::STEP]
            keep = d.label.to_numpy() < 3
            Xs.append(d[cols].to_numpy(np.float32)[keep])
            ys.append(d.label.to_numpy()[keep])
            ws.append(np.full(keep.sum(), met_weight))
    for mid in dfl.MATCHES:
        tlab = None if target == "touch" else dfl.labels_variant(mid, dfl.load(mid), target)
        for s in (seed, seed + 1):
            d = frame(mid, s, miss, variant in ("extra", "both"), pools)
            if tlab is not None:
                d = d.copy()
                d["label"] = tlab
            w = possession_weights(d.label.to_numpy())
            d = d.iloc[::STEP]
            keep = d.label.to_numpy() < 3
            Xs.append(d[cols].to_numpy(np.float32)[keep])
            ys.append(d.label.to_numpy()[keep])
            ws.append(w[::STEP][keep] if variant in ("weighted", "both") else np.ones(keep.sum()))
    w = np.concatenate(ws) if (variant in ("weighted", "both") or metrica) else None
    mdl = make_model().fit(np.vstack(Xs), np.concatenate(ys), sample_weight=w)
    joblib.dump(mdl, path)
    print("  fitted " + os.path.basename(path), flush=True)
    return mdl


def team_probs(models, cols, miss, extra):
    """Mean P over `models` for each Metrica unit and each half of our match."""
    out = {"Metrica": [], "match": None}
    for g in ("M1", "M2", "M3"):
        fr = frame(g, 0, miss, extra)
        p = np.mean([m.predict_proba(fr[cols].to_numpy(np.float32)) for m in models], axis=0)
        out["Metrica"] += [p[s:e] for s, e in pm.valid_runs(fr.label.to_numpy())]
    ours, _ = features.build()
    if extra:
        ours = pd.concat([ours, extra_features("ours", 0, miss)], axis=1)
    out["match"] = np.mean([m.predict_proba(ours[cols].to_numpy(np.float32)) for m in models],
                           axis=0)
    return out


def with_team(S, names, probs):
    T = subset(S, names, [names[0]])
    T.units = {"Metrica": [(lab, [p], ps, pc) for (lab, _, ps, pc), p in
                           zip(T.units["Metrica"], probs["Metrica"])],
               "match": [(a, b, [probs["match"][a:b]], ps, pc)
                         for a, b, _, ps, pc in T.units["match"]]}
    return T


def source_probs(models, cols, miss, extra):
    """Full-length P of the mean of `models`: {"M1", "M2", "M3", "ours"}."""
    out = {}
    for g in ("M1", "M2", "M3"):
        fr = frame(g, 0, miss, extra)
        out[g] = np.mean([m.predict_proba(fr[cols].to_numpy(np.float32)) for m in models], axis=0)
    ours, _ = features.build()
    if extra:
        ours = pd.concat([ours, extra_features("ours", 0, miss)], axis=1)
    out["ours"] = np.mean([m.predict_proba(ours[cols].to_numpy(np.float32)) for m in models],
                          axis=0)
    return out


def with_sources(S, names, sources, miss):
    """Team from several full-length sources (each counted once in the home-share mean)."""
    T = subset(S, names, [names[0]])
    i, units = 0, []
    for g in ("M1", "M2", "M3"):
        lab_all = dfl_features(g, 0, miss).label.to_numpy()
        for s, e in pm.valid_runs(lab_all):
            lab, _, ps, pc = T.units["Metrica"][i]
            units.append((lab, [src[g][s:e] for src in sources], ps, pc))
            i += 1
    T.units = {"Metrica": units,
               "match": [(a, b, [src["ours"][a:b] for src in sources], ps, pc)
                         for a, b, _, ps, pc in T.units["match"]]}
    return T


TCNS = ("dfl_tcn_all", "dfl_tcn_all_s1", "dfl_tcn_all_s2", "dfl_tcn_v_dil128", "dfl_tcn_v_ep12",
        "dfl_tcn_v_st", "dfl_tcn_v_ch128", "dfl_tcn_v_extra")


def bag(S, names, miss, base_cols, xcols):
    pools = state_pools()
    cols = base_cols + xcols
    models = ([fit("extra", s, miss, cols) for s in (0, 2, 4, 6, 8, 10)] +
              [fit("extra", s, miss, cols, pools) for s in (0, 2, 4)])
    hgb_x = [source_probs([m], cols, miss, True) for m in models]
    hgb = [source_probs([m], base_cols, miss, False) for m in vc.load_models(vc.PLAIN + vc.STATE)]
    nets = []
    for st in TCNS:
        d = {g: np.load(os.path.join(data.CACHE, "{}_{}.npy".format(st, g))).astype(np.float64)
             for g in ("M1", "M2", "M3")}
        d["ours"] = np.load(os.path.join(data.CACHE, st + ".npy")).astype(np.float64)
        nets.append(d)

    def mean_source(srcs):
        return {g: np.mean([s[g] for s in srcs], axis=0) for g in srcs[0]}

    def weighted(a, b, w):
        """Home-share mixture: bag a with weight 1 - w, bag b with weight w."""
        out = {}
        for g in a[0]:
            ha = np.mean([s[g][:, 0] / np.maximum(s[g][:, 0] + s[g][:, 1], 1e-9) for s in a], 0)
            hb = np.mean([s[g][:, 0] / np.maximum(s[g][:, 0] + s[g][:, 1], 1e-9) for s in b], 0)
            h = (1 - w) * ha + w * hb
            out[g] = np.column_stack([h, 1 - h, np.zeros_like(h)])
        return out

    ref = vc.evaluate(with_sources(S, names, hgb, miss), FULL)
    vc.report("submission: 9 boosting models", ref, ref)
    rows = [("9 boosting models with the new features", hgb_x)]
    for w in (0.25, 0.5, 0.75):
        rows.append(("9 boosting + 8 networks, networks {:.2f}".format(w),
                     [weighted(hgb, nets, w)]))
        rows.append(("9 boosting (new features) + 8 networks, networks {:.2f}".format(w),
                     [weighted(hgb_x, nets, w)]))
    rows.append(("8 networks alone", [mean_source(nets)]))
    for name, srcs in rows:
        res = vc.evaluate(with_sources(S, names, srcs, miss), FULL)
        vc.report(name, ref, res)
        print("    official: " + official_paired(ref, res), flush=True)
    best = [weighted(hgb_x, nets, 0.5)]
    for tc, bonus in ((20.0, 16.0), (36.0, 30.0), (48.0, 39.0)):
        res = vc.evaluate(with_sources(S, names, best, miss), dict(FULL, team=tc, bonus=bonus))
        vc.report("  new features + networks 0.5, team cost {:.0f}/{:.0f}".format(tc, bonus),
                  ref, res)
        print("    official: " + official_paired(ref, res), flush=True)


BOOST = [(s, False) for s in (0, 2, 4, 6, 8, 10)] + [(s, True) for s in (0, 2, 4)]


def write_probas(miss, base_cols, xcols):
    """Each of the 9 boosting team models with the new features (6 plain, 3 state-noise), on our
    match: cache/team_extra[_st]_s<seed>_ours.npy (scripts/make_submission.py), and on the
    Metrica games: ..._M1.npy etc."""
    pools = state_pools()
    cols = base_cols + xcols
    ours, _ = features.build()
    X = pd.concat([ours, extra_features("ours", 0, miss)], axis=1)[cols].to_numpy(np.float32)
    for seed, st in BOOST:
        mdl = fit("extra", seed, miss, cols, pools if st else None)
        stem = "team_extra{}_s{}".format("_st" if st else "", seed)
        np.save(os.path.join(data.CACHE, stem + "_ours.npy"),
                mdl.predict_proba(X).astype(np.float32))
        for g in ("M1", "M2", "M3"):          # the Metrica validation games (scripts/team_mix.py)
            fr = frame(g, 0, miss, True)
            np.save(os.path.join(data.CACHE, "{}_{}.npy".format(stem, g)),
                    mdl.predict_proba(fr[cols].to_numpy(np.float32)).astype(np.float32))
        print("wrote cache/{}_*.npy".format(stem), flush=True)


def main():
    warnings.filterwarnings("ignore")
    miss = our_missing_mask()
    ours, _ = features.build()
    base_cols = [c for c in ours.columns if not re.search(DROP, c)]
    xcols = list(extra_features("ours", 0, miss).columns)
    if sys.argv[1:] == ["proba"]:
        return write_probas(miss, base_cols, xcols)
    names = list(vc.PLAIN[:1])
    dead = [joblib.load(os.path.join(data.CACHE, "pff_dflpff_st_s{}.joblib".format(s)))
            for s in (0, 2, 4)]
    S = vc.Samples(plain=vc.load_models(names), state=dead)
    set_pdc(S, detector("pff", miss, base_cols), miss, base_cols, ours)
    if sys.argv[1:] == ["bag"]:
        return bag(S, names, miss, base_cols, xcols)
    ref_models = vc.load_models(["dfl_model.joblib", "dfl_model_s2.joblib"])
    ref = vc.evaluate(with_team(S, names, team_probs(ref_models, base_cols, miss, False)), FULL)
    vc.report("reference: 2 plain DFL models", ref, ref)
    for variant in VARIANTS:
        extra = variant in ("extra", "both")
        cols = base_cols + (xcols if extra else [])
        models = [fit(variant, s, miss, cols) for s in SEEDS]
        res = vc.evaluate(with_team(S, names, team_probs(models, cols, miss, extra)), FULL)
        vc.report(variant, ref, res)
        print("    official: " + official_paired(ref, res), flush=True)


if __name__ == "__main__":
    main()
