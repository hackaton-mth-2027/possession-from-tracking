"""Full-match sanity checks of a submission, without labels: the traps the 5 public minutes
cannot show (the pre-kickoff dead ball was one).

    .venv/bin/python tools/sanity_full_match.py [outputs/submissions/dfl_bag_st_dflcd.csv]

1. Long collective pauses (players' mean speed < 1 m/s for >= 20 s: injuries, VAR, cooling
   breaks, goal celebrations): share predicted dead ball.
2. Goals (Smart Tagging): dead ball between the goal and the kickoff that follows.
3. Set pieces (Smart Tagging restarts): a dead ball in the 10 s before the tagged restart.
4. Possession durations: very short and very long ones; long dead-ball runs.
5. Per 15-minute block: possessions per live minute and dead-ball share (consistency).
Counts only.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
from possession import data  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = data.FPS


def runs_of(mask):
    m = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "outputs", "submissions", "dfl_bag_st_dflcd.csv")
    lab = pd.read_csv(path)["label"].astype(str).to_numpy()
    graded = np.zeros(len(lab), bool)
    for _, a, b in data.halves():
        graded[a:b + 1] = True
    t = data.load_tracking()
    speed = np.nanmean(t["player_speed"], axis=1)
    speed_s = pd.Series(speed).rolling(FPS, center=True, min_periods=1).mean().to_numpy()

    print("1. long collective pauses (mean player speed < 1 m/s for >= 20 s)")
    pauses = [(a, b) for a, b in runs_of(graded & (speed_s < 1.0)) if b - a >= 20 * FPS]
    for a, b in pauses:
        d = np.mean(lab[a:b] == "D")
        flag = "" if d > 0.8 else "   <-- check"
        print("   frames {:>6}-{:<6} {:5.0f} s   predicted dead {:4.0%}{}".format(
            a, b, (b - a) / FPS, d, flag))

    tags = data.load_tags()
    print("\n2. goals (Smart Tagging) -> next kickoff")
    goals = tags[tags.code == "GOALS"]
    kick = tags[(tags.code == "SET PIECES") & tags.type.astype(str).str.lower().str.contains("kick off")]
    for g in goals.itertuples():
        nxt = kick[kick.start > g.start]
        if not len(nxt):
            continue
        k = int(nxt.start.iloc[0])
        a = int(g.end) if g.end < k else int(g.start)
        print("   goal tag {}-{} -> kickoff {}: dead ball over the gap {:4.0%} ({:.0f} s)".format(
            int(g.start), int(g.end), k, np.mean(lab[a:k] == "D"), (k - a) / FPS))

    print("\n3. set pieces (Smart Tagging): dead ball in the 10 s before the tagged restart")
    sp = tags[(tags.code == "SET PIECES") & graded[np.clip(tags.start.astype(int), 0,
                                                            len(lab) - 1)]]
    rows = []
    for r in sp.itertuples():
        s = int(r.start)
        rows.append((str(r.type), np.any(lab[max(0, s - 10 * FPS):s + FPS] == "D")))
    df = pd.DataFrame(rows, columns=["type", "dead_before"])
    print(df.groupby("type").dead_before.agg(["size", "mean"]).round(2).to_string())

    print("\n4. durations")
    codes = pm.label_codes(lab)
    halves = [(a, b + 1) for _, a, b in data.halves()]
    sp_ = pm.possession_spells(codes, halves)
    d = np.array([(b - a) / FPS for a, b, _ in sp_])
    print("   {} possessions: < 1 s {}, < 2 s {}, > 60 s {}, longest {:.0f} s".format(
        len(d), int((d < 1).sum()), int((d < 2).sum()), int((d > 60).sum()), d.max()))
    dead = [(a, b) for a, b, k in pm.segments(codes, halves) if k == pm.D]
    dl = np.array([(b - a) / FPS for a, b in dead])
    print("   {} dead-ball runs: median {:.0f} s, > 60 s {}, longest {:.0f} s".format(
        len(dl), np.median(dl), int((dl > 60).sum()), dl.max()))
    for a, b in dead:
        if (b - a) / FPS > 60:
            print("     long dead ball {}-{} ({:.0f} s): mean player speed {:.1f} m/s, "
                  "ball seen {:.0%}".format(a, b, (b - a) / FPS, np.nanmean(speed[a:b]),
                                           np.mean(~np.isnan(t["ball"][a:b, 0]))))

    print("\n5. per 15-minute block")
    print("   block         poss/live-min   dead share   home share")
    for _, a, b in data.halves():
        for s in range(a, b + 1, 15 * 60 * FPS):
            e = min(b + 1, s + 15 * 60 * FPS)
            w = codes[s:e]
            live = np.isin(w, [pm.H, pm.A])
            n = len(pm.possession_spells(w, [(0, e - s)]))
            print("   {:>6}-{:<6}   {:10.2f}   {:10.0%}   {:10.0%}".format(
                s, e, n / max(1e-9, live.sum() / FPS / 60), np.mean(w == pm.D),
                np.mean(w[live] == pm.H) if live.any() else np.nan))


if __name__ == "__main__":
    main()
