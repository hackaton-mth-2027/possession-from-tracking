"""A validation set with REAL automatic tracking: SkillCorner's 20 open A-League matches.

    .venv/bin/python scripts/skillcorner_val.py [--rule lite|events|strict] [--matches 20]
        [--flat] [--only NAME[,NAME...]]

The Metrica validation set (scripts/metrica_val.py, validate_clean.py) is clean tracking degraded
by possession/dfl.add_noise, the simulator that also made the DFL training copies: it cannot tell
whether a choice holds on noise we did not write. Here the tracking is SkillCorner's broadcast
tracking as provided (possession/skillcorner.py; 10 fps resampled to 25 Hz, no noise added), and
the truth comes from SkillCorner's own player possessions and phases of play, under the same
controlled-possession rule as the Metrica labels (`--rule`). Nothing is trained on it.

The one transformation: airborne balls are drawn where our camera's line of sight meets the
ground (as dfl.add_noise does first, from SkillCorner's ball height), because that is how our
match's ball is tracked; `--flat` leaves the ball as provided.

Every member of the submission is applied to each match from its saved model (make_submission.py:
9 boosting team models, 16 networks, 3 dead-ball models, change detector, dead-ball boundary
detector) and decoded per half with the submission's costs. Variants change one thing at a time:
the families' weights, the decoder's costs, and the team models that the Metrica set rejected
(PFF-trained boosting models and networks, networks on DFL's rule-based labels). Reported: spellF1
(the official scorer) pooled over the matches' possessions, the paired difference with the
submission with a 90% interval over possessions, and in how many matches the variant is better.
Members cached per match as cache/sc_members_<gid>[_flat].npz; aggregates only are printed.
"""

import argparse
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
from possession import data, decode, dfl, features, features_extra  # noqa: E402
from possession import skillcorner as sc  # noqa: E402
import possession_metrics as pm  # noqa: E402
import change_detector as cd  # noqa: E402
import make_submission as ms  # noqa: E402
from compare import flags_and_preds, f1  # noqa: E402
from dfl_pretrain import DROP  # noqa: E402

BOOST = ["team_extra{}_s{}".format("_st" if st else "", s) for s, st in ms.BOOST]
NETS = list(ms.NETS)
MET_NETS = list(ms.MET_NETS)
DEAD = ["pff_dflpff_st_s{}".format(s) for s in ms.PFF_SEEDS]
# team models the Metrica set rejected (TECHNICAL.md, section 3): boosting on PFF or DFL + PFF, networks
# with PFF, networks on DFL's rule-based labels; and the first 9 boosting models (no extra features)
ALT_BOOST = ["pff_pffonly_plain_s0", "pff_pffonly_plain_s2", "pff_dflpff_plain_s0",
             "pff_dflpff_plain_s2"] + ["dfl_model"] + ["dfl_model_s{}".format(s) for s in
                                                       (2, 4, 6, 8, 10)]
ALT_NETS = ["dfl_tcn_pffl02", "dfl_tcn_pffl10", "dfl_tcn_pffe02", "dfl_tcn_dlite",
            "dfl_tcn_dlite_s1", "dfl_tcn_dlite_s2", "dfl_tcn_dstrict", "dfl_tcn_dstrict_s1",
            "dfl_tcn_dstrict_s2", "dfl_tcn_dstrict_s3", "dfl_tcn_dstrict_s4"]


def frames(gid, flat=False):
    """Base and extra features of a SkillCorner match as tracked, plus its "events" labels.
    Cached as cache/sc_feat_<gid>[_flat].pkl."""
    path = os.path.join(data.CACHE, "sc_feat_{}{}.pkl".format(gid, "_flat" if flat else ""))
    if os.path.exists(path):
        return pd.read_pickle(path)
    m = sc.load(gid)
    if not flat:
        m = dict(m)
        k = dfl.CAMERA[2] / np.maximum(dfl.CAMERA[2] - m["ball_z"], 1e-3)
        m["ball"] = (dfl.CAMERA[:2] + k[:, None] * (m["ball"] - dfl.CAMERA[:2])).astype(np.float32)
    df, _ = features.build(match=m)
    df = pd.concat([df, features_extra.build_extra(m)], axis=1)
    df["label"] = m["labels"]
    df.to_pickle(path)
    return df


def home_share(p):
    return (p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-9)).astype(np.float32)


def members(gid, base_cols, xcols, flat=False):
    """{member: home share (team models), P(dead) ("dead:" + name), "pc", "pdc", "half"} of one
    match, each from its saved model."""
    path = os.path.join(data.CACHE, "sc_members_{}{}.npz".format(gid, "_flat" if flat else ""))
    out = dict(np.load(path)) if os.path.exists(path) else {}
    todo = [n for n in BOOST + ALT_BOOST + NETS + MET_NETS + ALT_NETS if n not in out] + \
        [n for n in DEAD if "dead:" + n not in out] + \
        [k for k in ("pc", "pdc", "blind") if k not in out]
    if not todo:
        return out
    from dfl_change import inputs
    from dead_change import dead_inputs
    from dfl_tcn import load_net, predict, prepare
    df = frames(gid, flat)
    out["half"] = df["half"].to_numpy().astype(np.int8)
    out["blind"] = (df["home_n"].to_numpy() + df["away_n"].to_numpy()) == 0
    Xb = df[base_cols].to_numpy(np.float32)
    Xx = df[base_cols + xcols].to_numpy(np.float32)
    for n in todo:
        if n in ("pc", "pdc", "blind"):
            continue
        present = os.path.exists(os.path.join(data.CACHE, n + ".joblib")) or \
            os.path.exists(os.path.join(data.CACHE, n + ".pt"))
        if not present:
            continue
        if n in NETS + MET_NETS + ALT_NETS:
            net, cols, mean, std = load_net(n)
            out[n] = home_share(predict(net, prepare(df, cols, mean, std)))
        else:
            mdl = joblib.load(os.path.join(data.CACHE, n + ".joblib"))
            p = mdl.predict_proba(Xx if n in BOOST else Xb)
            if n in DEAD:
                out["dead:" + n] = p[:, 2].astype(np.float32)
            else:
                out[n] = home_share(p)
    if "pc" not in out:
        det, dcols = joblib.load(os.path.join(data.CACHE, "dfl_change_touch.joblib"))
        out["pc"] = det.predict_proba(inputs(df, dcols))[:, 1].astype(np.float32)
    if "pdc" not in out:
        det = joblib.load(os.path.join(data.CACHE, "dchange_pff.joblib"))
        out["pdc"] = det.predict_proba(dead_inputs(df, base_cols))[:, 1].astype(np.float32)
    np.savez(path, **out)
    return out


def viterbi(proba, pc, pdc, half, team=ms.TEAM_COST, bonus=ms.BONUS, dead=ms.DEAD_COST,
            dead_bonus=ms.DEAD_BONUS):
    """make_submission.decode_steps (frame by frame) per half, with the costs as arguments."""
    out = np.full(len(proba), pm.X, np.int8)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        p = np.clip(proba[idx], 1e-6, 1.0)
        tc = np.maximum(cd.MIN_COST, team - bonus * pc[idx])
        dd = np.maximum(cd.MIN_COST, dead - dead_bonus * pdc[idx])
        out[idx] = decode.viterbi_tv(p / p.sum(1, keepdims=True), tc, dd, dd, eps=1e-300)
    return out


def predict_match(M, boost=BOOST, nets=NETS + MET_NETS, w=ms.NET_WEIGHT, blind=True,
                  true_dead=None, **costs):
    """`blind`: where SkillCorner tracks no player at all (the broadcast shows a replay or a
    close-up, in stretches of 5 s or more: 55% of a match's dead-ball frames, 1.7% of its live
    ones) the frame is given to the decoder as dead ball, and main() leaves it out of the
    grading. Our tactical camera never goes blind, and the models have never seen a frame
    without players. `true_dead`: the labels' dead ball instead of the models' (the team estimate
    alone)."""
    fam = [np.mean([M[n] for n in names], axis=0) for names in (boost, nets) if names]
    home = fam[0] if len(fam) == 1 else (1 - w) * fam[0] + w * fam[1]
    d = np.mean([M["dead:" + n] for n in DEAD], axis=0)
    if blind:
        d = np.where(M["blind"], 1.0, d)
    if true_dead is not None:
        d = np.where(true_dead, 1.0 - 1e-6, 1e-6)
    proba = np.column_stack([(1 - d) * home, (1 - d) * (1 - home), d]).astype(np.float64)
    return viterbi(proba, M["pc"].astype(np.float64), M["pdc"].astype(np.float64), M["half"],
                   **costs)


def variants(have):
    pff_b = [n for n in ALT_BOOST if n.startswith("pff_pffonly") and n in have]
    mix_b = [n for n in ALT_BOOST if n.startswith("pff_dflpff") and n in have]
    old_b = [n for n in ALT_BOOST if n.startswith("dfl_model") and n in have]
    pff_n = [n for n in ALT_NETS if "pff" in n and n in have]
    lite_n = [n for n in ALT_NETS if "dlite" in n and n in have]
    strict_n = [n for n in ALT_NETS if "dstrict" in n and n in have]
    both = NETS + MET_NETS
    v = {"submission (9 boosting + 16 networks, 28/23)": {}}
    v["  boosting alone"] = dict(nets=[])
    v["  networks alone"] = dict(boost=[])
    v["  networks weight 0.25"] = dict(w=0.25)
    v["  networks weight 0.75"] = dict(w=0.75)
    v["  DFL networks only (no Metrica-trained)"] = dict(nets=NETS)
    v["  first boosting models (no extra features)"] = dict(boost=old_b)
    v["  no change detector (bonus 0)"] = dict(bonus=0.0)
    for tc, bn in ((16.0, 13.0), (20.0, 16.0), (36.0, 30.0), (48.0, 39.0), (64.0, 52.0)):
        v["  team cost {:.0f}/{:.0f}".format(tc, bn)] = dict(team=tc, bonus=bn)
    for w, tc in ((0.75, 36.0), (0.75, 48.0), (0.75, 64.0), (1.0, 48.0)):
        v["  networks weight {:.2f}, team cost {:.0f}/{:.0f}".format(w, tc, 0.82 * tc)] = \
            dict(w=w, team=tc, bonus=round(0.82 * tc, 1), **({"boost": []} if w == 1.0 else {}))
    v["  boosting = PFF-trained (2)"] = dict(boost=pff_b)
    v["  boosting = DFL + PFF-trained (2)"] = dict(boost=mix_b)
    v["  boosting + PFF-trained (9 + 2)"] = dict(boost=BOOST + pff_b)
    v["  boosting + PFF and DFL + PFF (9 + 4)"] = dict(boost=BOOST + pff_b + mix_b)
    v["  networks + PFF networks (16 + 3)"] = dict(nets=both + pff_n)
    v["  networks = PFF networks (3)"] = dict(nets=pff_n)
    v["  networks + DFL lite-label networks"] = dict(nets=both + lite_n)
    v["  networks + DFL strict-label networks"] = dict(nets=both + strict_n)
    v["  networks = DFL strict-label networks"] = dict(nets=strict_n)
    v["  all PFF: boosting PFF, networks PFF"] = dict(boost=pff_b, nets=pff_n)
    return {k: a for k, a in v.items() if all(len(a.get(f, [1])) for f in ("boost", "nets"))
            or a.get("nets") == [] or a.get("boost") == []}


def check(games):
    """The labels against physics and against the official window's possession structure."""
    rows = {r: [] for r in ("events", "lite", "strict")}
    phys = []
    for gid in games:
        m, r = sc.load(gid), sc.raw(gid)
        for rule in rows:
            lab = sc.labels(gid, rule)
            d = np.array([(e - s) / data.FPS for s, e, _ in
                          pm.possession_spells(lab, pm.valid_runs(lab))])
            rows[rule].append((len(d), np.mean(d < 2), np.mean(d < 5), np.median(d), d.mean(),
                               len(d) / (np.isin(lab, (0, 1)).sum() / data.FPS / 60),
                               np.sum(lab == 2) / np.sum(lab < 3)))
        lab = sc.labels(gid, "lite")
        live, dead = np.isin(lab, (0, 1)), lab == 2
        seen = ~np.isnan(m["ball"][:, 0])
        blind = np.isnan(m["players"][:, :, 0]).all(axis=1)
        sp = np.nanmedian(m["player_speed"], axis=1)
        ev = sc.events(gid)
        slot = {p["id"]: k for k, p in enumerate(sc.metadata(gid)["players"])}
        near = []
        for a, b, pid in zip(sc._index(r, ev.frame_start), sc._index(r, ev.frame_end),
                             ev.player_id):
            if pid not in slot:           # 1 event of one match names a player not in its metadata
                continue
            dist =np.linalg.norm(r["players"][a:b + 1, slot[pid]] - r["ball"][a:b + 1, :2], axis=1)
            near.append(dist[r["ball_det"][a:b + 1]])
        near = np.concatenate(near)
        phys.append((seen[live].mean(), seen[dead].mean(), np.nanmedian(sp[live & ~blind]),
                     np.nanmedian(sp[dead & ~blind]), blind[live].mean(), blind[dead].mean(),
                     np.nanmedian(near), np.nanmean(near < 1.5),
                     ev.is_player_possession_start_matched.mean()))
    print("{} matches. Official window: < 2 s 12%, < 5 s 35%, median 7.6 s, mean 14.2 s, 4.2 "
          "per live minute, dead ball 20%".format(len(games)))
    for rule, v in rows.items():
        v = np.array(v)
        print("{:7s} {} possessions, < 2 s {:.0%}, < 5 s {:.0%}, median {:.1f} s, mean {:.1f} s, "
              "{:.1f} per live minute (matches {:.1f}-{:.1f}), dead ball {:.0%}".format(
                  rule, int(v[:, 0].sum()), *np.average(v[:, 1:5], axis=0, weights=v[:, 0]),
                  v[:, 5].mean(), v[:, 5].min(), v[:, 5].max(), v[:, 6].mean()))
    p = np.array(phys).mean(0)
    print("ball detected on {:.0%} of live frames, {:.0%} of dead ones; median player speed "
          "{:.2f} m/s live, {:.2f} dead; no player tracked on {:.1%} of live frames, {:.0%} of "
          "dead ones".format(*p[:6]))
    print("player in possession to the ball: median {:.2f} m, within 1.5 m {:.0%}; possessions "
          "matched to event data {:.0%}".format(*p[6:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rule", default="lite", choices=("lite", "events", "strict"))
    ap.add_argument("--matches", type=int, default=20)
    ap.add_argument("--flat", action="store_true", help="the ball as provided (no projection)")
    ap.add_argument("--only", default="", help="substrings of the variants to run")
    ap.add_argument("--sweep", action="store_true",
                    help="the main families over the team-switch cost: spellF1 (predicted)")
    ap.add_argument("--check", action="store_true",
                    help="the labels against physics and the official window's structure")
    ap.add_argument("--true-dead", action="store_true",
                    help="dead ball from the labels: the team estimate and the decoder alone")
    ap.add_argument("--no-blind", action="store_true",
                    help="do not give frames without any tracked player as dead ball")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    games = sc.games()[:args.matches]
    if args.check:
        return check(games)
    ours, _ = features.build()
    base_cols = [c for c in ours.columns if not re.search(DROP, c)]
    del ours
    xcols = None
    per = {}
    for gid in games:
        if xcols is None:
            xcols = [c for c in frames(gid, args.flat).columns
                     if c not in base_cols and not re.search(DROP, c) and c != "label"]
        M = members(gid, base_cols, xcols, args.flat)
        gt = sc.labels(gid, args.rule)
        if not args.no_blind:                 # nothing tracked: given as dead ball, not graded
            gt = np.where(M["blind"], pm.X, gt).astype(gt.dtype)
        per[gid] = (gt, M)
    have = set.intersection(*[set(M) for _, M in per.values()])
    V = variants(have)
    if args.only:
        keep = [s for s in args.only.split(",")]
        V = {k: a for i, (k, a) in enumerate(V.items()) if i == 0 or any(s in k for s in keep)}

    def run(kw):
        fl, npd, per_match, stats = [], [], [], []
        for gid in games:
            gt, M = per[gid]
            pred = np.where(gt == pm.X, pm.X, predict_match(
                M, blind=not args.no_blind, true_dead=(gt == pm.D) if args.true_dead else None,
                **kw))
            runs = pm.valid_runs(gt)
            f, n = flags_and_preds(gt, pred, runs)
            fl.append(f)
            npd.append(n)
            per_match.append(f1(f, n, np.arange(f.shape[1])))
            live = np.isin(gt, (pm.H, pm.A))
            stats.append((f.shape[1], n.sum(), np.mean(pred[live] == gt[live]),
                          np.sum((pred == pm.D) & (gt == pm.D)), np.sum(pred == pm.D),
                          np.sum(gt == pm.D), np.mean(pred[live & (pred != pm.D)] ==
                                                      gt[live & (pred != pm.D)])))
        return np.hstack(fl), np.concatenate(npd), np.array(per_match), np.array(stats)

    if args.sweep:
        # is a variant better than the submission at the same number of possessions, or does
        # it only split less? Each family over the team-switch cost (bonus 0.82 x the cost).
        rows = {k: V[k] for k in V if "team cost" not in k and any(s in k for s in (
            "submission", "boosting = PFF", "9 + 4", "networks = PFF", "strict-label networks",
            "networks alone", "weight 0.75"))}
        rows["  no change detector"] = dict(bonus=None)
        costs = (16.0, 20.0, 28.0, 36.0, 48.0, 64.0, 80.0)
        print("{:<46s} ".format("team cost") + " ".join("{:^12.0f}".format(c) for c in costs))
        for name, kw in rows.items():
            cells = []
            for c in costs:
                bonus = 0.0 if kw.get("bonus", 1) is None else round(0.82 * c, 1)
                fl, npd, _, st = run(dict({k: a for k, a in kw.items() if k != "bonus"},
                                          team=c, bonus=bonus))
                cells.append("{:.3f} ({:4d})".format(f1(fl, npd, np.arange(fl.shape[1])),
                                                     int(st[:, 1].sum())))
            print("{:<46s} ".format(name) + " ".join(cells), flush=True)
        return
    r = np.random.RandomState(0)
    ref = None
    for name, kw in V.items():
        fl, npd, pmatch, st = run(kw)
        allk = np.arange(fl.shape[1])
        if ref is None:
            ref = (fl, npd, pmatch)
            draws = [r.randint(0, len(allk), len(allk)) for _ in range(1000)]
            print("SkillCorner, {} matches, labels {}{}: {} possessions; predicted {}; frame "
                  "accuracy on live frames {:.3f}, team where called live {:.3f}; dead ball "
                  "recall {:.2f} precision {:.2f}".format(
                      len(games), args.rule, (", ball as provided" if args.flat else "") +
                      (", true dead ball" if args.true_dead else "") +
                      (", untracked frames left to the models and graded" if args.no_blind
                       else ""),
                      int(st[:, 0].sum()), int(st[:, 1].sum()), st[:, 2].mean(), st[:, 6].mean(),
                      st[:, 3].sum() / st[:, 5].sum(), st[:, 3].sum() / st[:, 4].sum()))
            print("per match spellF1: " + " ".join("{:.2f}".format(x) for x in pmatch))
        d = np.array([f1(fl, npd, k) - f1(ref[0], ref[1], k) for k in draws])
        print("{:<46s} spellF1 {:.3f} {:+.3f} [{:+.3f}, {:+.3f}]  better in {:2d}/{} matches  "
              "predicted {}".format(name, f1(fl, npd, allk), f1(fl, npd, allk) -
                                    f1(ref[0], ref[1], allk), *np.percentile(d, [5, 95]),
                                    int(np.sum(pmatch > ref[2] + 1e-9)), len(games),
                                    int(st[:, 1].sum())), flush=True)


if __name__ == "__main__":
    main()
