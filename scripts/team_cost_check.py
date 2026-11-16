"""The submission's team-switch cost and network weight on the Metrica validation set, for the
settings scripts/skillcorner_val.py prefers on real tracking.

    METRICA_LABELS=lite .venv/bin/python scripts/team_cost_check.py

The submission's members from their caches (as scripts/hsmm_lite.py), decoded with the
submission's dead-ball costs; team cost c with bonus 0.82 c, and the networks' weight in the team
estimate. Paired on the Metrica set against the submission (28 / 23, weight 0.5). Also printed,
label-free: the possessions the setting gives over our match, per live minute.

    .venv/bin/python scripts/team_cost_check.py signal

How noisy is the team estimate before decoding, without labels: on frames the dead-ball models
call live (P(dead) < 0.5), how often the submission's home share crosses 0.5 per minute, how far
it is from 0 or 1 on average, and the share of frames between 0.3 and 0.7. For our match (whole,
official window's frames, the rest), the Metrica set (simulated noise) and the SkillCorner matches
(real broadcast tracking): which validation set does our match's signal look like?
"""

import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
import numpy as np  # noqa: E402
from possession import data  # noqa: E402
import validate_clean as vc  # noqa: E402
import make_submission as ms  # noqa: E402
from team_features import with_sources  # noqa: E402
from team_retest import FULL  # noqa: E402
from team_mix import mixed, samples, source  # noqa: E402


def raw_stats(pairs):
    """(crossings of 0.5 per live minute, mean distance to the nearer of 0 and 1, share of
    frames in 0.3-0.7, live minutes) over (home share, P(dead)) sequences."""
    flips = n = amb = mid = 0
    for h, d in pairs:
        live = d < 0.5
        x = h > 0.5
        flips += np.sum((x[1:] != x[:-1]) & live[1:] & live[:-1])
        n += live.sum()
        amb += np.minimum(h, 1 - h)[live].sum()
        mid += ((h > 0.3) & (h < 0.7))[live].sum()
    return flips / (n / data.FPS / 60), amb / n, mid / n, n / data.FPS / 60


def signal(S, names, miss, boost, nets):
    import skillcorner_val as sv
    from possession import skillcorner as sc
    T = with_sources(S, names, [mixed(boost, nets)], miss)
    rows = {}
    rows["Metrica set (simulated noise)"] = [
        (pp[0][:, 0], np.mean([p[:, 2] for p in ps], axis=0)) for _, pp, ps, _ in
        T.units["Metrica"]]
    gt = S.official
    whole, window, rest = [], [], []
    for a, b, pp, ps, _ in T.units["match"]:
        h, d = pp[0][:, 0], np.mean([p[:, 2] for p in ps], axis=0)
        whole.append((h, d))
        w = gt[a:b] != 3
        if w.any():
            window.append((h[w], d[w]))
        rest.append((h[~w], d[~w]))
    rows["our match"] = whole
    rows["  the official window's frames"] = window
    rows["  the rest of our match"] = rest
    sk = []
    for tag, what in (("", "airborne balls projected"), ("_flat", "ball as provided")):
        cur = []
        for gid in sc.games():
            path = os.path.join(data.CACHE, "sc_members_{}{}.npz".format(gid, tag))
            if not os.path.exists(path):
                continue
            M = np.load(path)
            h = 0.5 * np.mean([M[n] for n in sv.BOOST], 0) + \
                0.5 * np.mean([M[n] for n in sv.NETS + sv.MET_NETS], 0)
            d = np.where(M["blind"], 1.0, np.mean([M["dead:" + n] for n in sv.DEAD], 0))
            keep = M["half"] > 0
            cur.append((h[keep], d[keep]))
        if cur:
            rows["SkillCorner, {} ({} matches)".format(what, len(cur))] = cur
        sk = sk or cur
    for name, pairs in rows.items():
        print("{:<50s} {:5.1f} crossings per live minute, mean distance {:.3f}, in 0.3-0.7 "
              "{:.1%} ({:.0f} live minutes)".format(name, *raw_stats(pairs)))
    for name, pairs in (("our match, by 15-minute block", whole),):
        h = np.concatenate([p[0] for p in pairs])
        d = np.concatenate([p[1] for p in pairs])
        step = 15 * 60 * data.FPS
        print(name + ": " + " ".join("{:.1f}".format(raw_stats([(h[k:k + step], d[k:k + step])])[0])
                                     for k in range(0, len(h), step)))
    per = [raw_stats([p])[0] for p in sk]
    print("SkillCorner, by match: " + " ".join("{:.1f}".format(x) for x in per))


def main():
    warnings.filterwarnings("ignore")
    S, names, miss = samples()
    boost = [source("team_extra{}_s{}".format("_st" if st else "", s), "_ours")
             for s, st in ms.BOOST]
    nets = [source(st) for st in ms.NETS] + [source(st[:-len("_met3")] + "_lomo")
                                             for st in ms.MET_NETS]
    if sys.argv[1:] == ["signal"]:
        return signal(S, names, miss, boost, nets)
    ref = None
    for w in (0.5, 0.75, 1.0):
        T = with_sources(S, names, [mixed(boost, nets, w)], miss)
        for c in (28.0, 36.0, 48.0, 64.0, 80.0):
            res = vc.evaluate(T, dict(FULL, team=c, bonus=round(0.82 * c, 1)))
            ref = ref or res
            sc, diff, (lo, hi) = vc.paired(ref[0], res[0])
            print("networks {:.2f}, team cost {:.0f}/{:.0f}: Metrica {:.3f} {:+.3f} [{:+.3f}, "
                  "{:+.3f}] macro {:.3f} | our match {} possessions, {:.2f} per live "
                  "minute".format(w, c, 0.82 * c, sc, diff, lo, hi, res[4], res[2], res[3]),
                  flush=True)


if __name__ == "__main__":
    main()
