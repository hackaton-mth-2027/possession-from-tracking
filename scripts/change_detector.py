"""Change signals, direction-free, for the detector of possession changes (scripts/dfl_change.py,
which learns it on the DFL matches), and the decoder's lowest team-switch cost.

For each live frame: how much the "who has it" signals differ between the half-second after and
the half-second before (nearest team, distance balance, movement towards the ball, tactical tags),
and whether the nearest team has just switched and keeps the new team for a while. The decoder's
team-switch cost drops where a change is likely: max(MIN_COST, team cost - bonus * p_change).
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import data  # noqa: E402

NEAR = 6            # frames: a frame within this of a change is positive
FAR = 25            # frames: negatives are at least this far from any change
HALF = 12           # frames: before / after windows for the change signals
MIN_COST = 4.0     # the decoder's lowest team-switch cost


def runs(mask):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def window_shift(x, half):
    """mean over (t, t+half] minus mean over [t-half, t), NaN-tolerant."""
    s = pd.Series(x)
    before = s.shift(1).rolling(half, min_periods=1).mean()
    after = s[::-1].shift(1).rolling(half, min_periods=1).mean()[::-1]
    return (after - before).to_numpy()


def change_signals(f, vision=False):
    out = {}
    cols = ["nearest_is_home", "d_diff", "ctl_chase_diff", "tag_attack", "ctl_team_run",
            "n_home_near", "n_away_near", "ball_x_home"]
    if vision:
        cols += ["vis_nearest_is_home", "vis_d_diff", "vis_ctl_chase_diff", "vis_ctl_team_run",
                 "vis_n_home_near", "vis_n_away_near", "vis_touch_last", "vis_touch_bal_r12"]
    if "dfl_home" in f:
        cols += ["dfl_home"]
    for col in cols:
        out["chg_" + col] = np.abs(window_shift(f[col].to_numpy(), HALF))
    if vision:
        run = np.abs(f["vis_ctl_team_run"].to_numpy())
        ahead = np.abs(f["vis_ctl_team_ahead"].to_numpy())
        out["chg_vis_switch_and_hold"] = np.where(run <= 2 / data.FPS, ahead, 0.0)
    run = np.abs(f["ctl_team_run"].to_numpy())
    ahead = np.abs(f["ctl_team_ahead"].to_numpy())
    out["chg_switch_and_hold"] = np.where(run <= 2 / data.FPS, ahead, 0.0)
    for col in ("ball_seen", "gap_len_s", "walk_all_r12", "ball_speed", "dfl_spread_r12",
                "dfl_minority_r12", "dfl_flips_r12"):
        if col in f:
            out[col] = f[col].to_numpy()
    return pd.DataFrame(out)
