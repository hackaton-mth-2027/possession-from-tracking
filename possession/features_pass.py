"""Passes between teammates and interceptions, from the ball's flights. Per the challenge's rules a
completed pass is the clearest sign of control; a ball landing at an opponent is a turnover.

A flight: the ball faster than FLIGHT_SPEED for at least FLIGHT_MIN frames. Its releaser / receiver:
the nearest player within NEAR metres in the frames just before take-off / just after landing.
Same team: a completed pass by that team; different teams: an interception by the receiver's team.

Per-frame features, signed (+1 home, -1 away, 0 none), times capped at CAP seconds:
  pass_last_team / pass_last_s    the last completed pass landed, by whom, how long ago
  pass_next_team / pass_next_s    the next completed pass takes off, by whom, in how long
  pass_flight_team                inside a completed pass's flight: its team
  int_last_team / int_last_s      the last interception (receiver's team), how long ago
  int_next_team / int_next_s      the next interception, in how long
  pass_balance_r50                completed passes by home minus away within +-2 s
`build_pass(match, ball=None)` takes the dict of possession/dfl.load or data.load_tracking.
"""

import numpy as np
import pandas as pd

from .data import FPS
from .features import clean_ball

FLIGHT_SPEED = 8.0        # m/s
FLIGHT_MIN = 4            # frames
NEAR = 2.5                # m
LOOK = 3                  # frames before take-off / after landing searched for the player
CAP = 10.0                # s


def flights(match, ball=None):
    """[(take-off frame, landing frame, releaser team, receiver team)], teams 0 / 1 / -1."""
    players, team = match["players"], match["team"]
    _, ball_i, _ = clean_ball(match["ball"] if ball is None else ball)
    pos = pd.DataFrame(ball_i).rolling(5, center=True, min_periods=3).mean().to_numpy()
    speed = np.linalg.norm(np.gradient(pos, axis=0), axis=1) * FPS
    fast = np.nan_to_num(speed) > FLIGHT_SPEED
    edges = np.flatnonzero(np.diff(np.r_[0, fast.astype(int), 0]))
    out = []
    n = len(ball_i)

    def nearest_team(frames):
        best, who = np.inf, -1
        for f in frames:
            if 0 <= f < n and not np.isnan(ball_i[f, 0]):
                d = np.linalg.norm(players[f] - ball_i[f], axis=1)
                if np.any(~np.isnan(d)):
                    k = int(np.nanargmin(d))
                    if d[k] < best:
                        best, who = d[k], int(team[k])
        return who if best <= NEAR else -1
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < FLIGHT_MIN:
            continue
        out.append((a, b, nearest_team(range(a - LOOK, a + 1)), nearest_team(range(b, b + LOOK))))
    return out


def _last(events, n, name):
    """Team (+1 / -1) and seconds since the last of `events` [(frame, sign)], at or before."""
    idx = np.arange(n)
    if not events:
        return {name + "_last_team": np.zeros(n), name + "_last_s": np.full(n, CAP)}
    frames = np.array([e[0] for e in events], np.int64)
    teams = np.array([e[1] for e in events])
    i = np.searchsorted(frames, idx, side="right") - 1
    ok = i >= 0
    i = np.maximum(i, 0)
    return {name + "_last_team": np.where(ok, teams[i], 0.0),
            name + "_last_s": np.where(ok, np.minimum((idx - frames[i]) / FPS, CAP), CAP)}


def _next(events, n, name):
    """Team and seconds until the next of `events`, at or after."""
    idx = np.arange(n)
    if not events:
        return {name + "_next_team": np.zeros(n), name + "_next_s": np.full(n, CAP)}
    frames = np.array([e[0] for e in events], np.int64)
    teams = np.array([e[1] for e in events])
    j = np.searchsorted(frames, idx, side="left")
    ok = j < len(frames)
    j = np.minimum(j, len(frames) - 1)
    return {name + "_next_team": np.where(ok, teams[j], 0.0),
            name + "_next_s": np.where(ok, np.minimum((frames[j] - idx) / FPS, CAP), CAP)}


def build_pass(match, ball=None, keep=1.0, seed=0):
    """`keep` < 1: each detected flight kept with that probability (to match the detection rate
    of our tracking, which loses fast balls more often than clean data)."""
    n = len(match["players"])
    sign = {0: 1.0, 1: -1.0}
    fl = flights(match, ball)
    if keep < 1.0:
        rng = np.random.default_rng(seed)
        fl = [x for x in fl if rng.random() < keep]
    passes = [(a, b, sign[r]) for a, b, r, c in fl if r >= 0 and r == c]
    inter = [(b, sign[c]) for a, b, r, c in fl if r >= 0 and c >= 0 and r != c]
    f = {}
    f.update(_last([(b, s) for a, b, s in passes], n, "pass"))       # since it landed
    f.update(_next([(a, s) for a, b, s in passes], n, "pass"))       # until it takes off
    f.update(_last(inter, n, "int"))
    f.update(_next(inter, n, "int"))
    flight = np.zeros(n)
    marks = np.zeros(n)
    for a, b, s in passes:
        flight[a:b] = s
        marks[min(b, n - 1)] += s
    f["pass_flight_team"] = flight
    f["pass_balance_r50"] = pd.Series(marks).rolling(101, center=True, min_periods=1).sum()
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})
