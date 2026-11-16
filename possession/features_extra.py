"""Extra per-frame features for the team estimate, next to possession/features.py.

Three physical signals, all signed (+ favours the home side) and NaN where the ball is unknown:

  interception  where the ball will be over the next INT_HORIZON seconds (straight line, rolling
                friction) and which team's player can get there first: each player starts after
                a reaction time from where his current velocity takes him, then runs at
                RUN_SPEED. int_team: the team of the first player to reach the ball's path (0 if
                nobody within the horizon); int_time_s: when; int_margin_s: how much later the
                other team's best player would get there; int_now_diff_s: time-to-ball now,
                away minus home
  carrier       a player close to the ball and moving with it (velocity aligned) over +-CARRY_W
                frames: carry_home / carry_away (best player of each team, 0-1), carry_diff
  goalkeeper    each team's deepest player with the ball close and slow (catching, holding,
                placing it for a goal kick): gk_hold (+1 home keeper, -1 away keeper), and the
                ball's distance to each team's deepest player

`build_extra(match, ball=None, hsign=None)` takes the dict of possession/dfl.load or
data.load_tracking (+ half, kickoff) and returns a DataFrame aligned with features.build's.
"""

import numpy as np
import pandas as pd

from .data import AWAY, FPS, HOME, PITCH_L, PITCH_W
from .features import ROLL, _rolling, clean_ball

# Params
RUN_SPEED = 7.0           # m/s: a player's top speed towards the ball
REACT = 0.3               # s: reaction time before he changes direction
FRICTION = 2.5            # m/s^2: a rolling ball's deceleration
INT_HORIZON = 2.0         # s: how far ahead the ball's path is followed
INT_STEP = 0.1            # s
CHUNK = 4000              # frames per vectorised chunk
CARRY_SCALE = 1.0         # m: distance scale of "close to the ball"
CARRY_W = (6, 12)         # frames: half-widths of the carrier windows
MOVING = 1.0              # m/s: below this, velocities are not compared
GK_NEAR = 1.5             # m
GK_SLOW = 2.0             # m/s


def _velocity(pos):
    """Velocity (m/s) from positions smoothed over 5 frames; pos (N, ..., 2)."""
    shape = pos.shape
    flat = pd.DataFrame(pos.reshape(shape[0], -1)).rolling(5, center=True, min_periods=3).mean()
    return np.gradient(flat.to_numpy().reshape(shape), axis=0) * FPS


def interception(ball_i, bvel, players, pvel, team):
    n, taus = len(ball_i), np.arange(0.0, INT_HORIZON + 1e-9, INT_STEP)
    out = {k: np.full(n, np.nan, np.float32) for k in
           ("int_team", "int_time_s", "int_margin_s", "int_now_diff_s")}
    home, away = team == HOME, team == AWAY
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        p, v = ball_i[a:b], np.nan_to_num(bvel[a:b])
        speed = np.linalg.norm(v, axis=1)
        unit = v / np.maximum(speed, 1e-6)[:, None]
        travel = np.minimum(speed[:, None] * taus - 0.5 * FRICTION * taus ** 2,
                            speed[:, None] ** 2 / (2 * FRICTION))                  # (c, K)
        q = p[:, None, :] + unit[:, None, :] * travel[:, :, None]                   # (c, K, 2)
        q[..., 0] = np.clip(q[..., 0], 0, PITCH_L)
        q[..., 1] = np.clip(q[..., 1], 0, PITCH_W)
        start = players[a:b] + np.nan_to_num(pvel[a:b]) * REACT                    # (c, P, 2)
        t_reach = REACT + np.linalg.norm(start[:, :, None, :] - q[:, None, :, :],
                                         axis=3) / RUN_SPEED                        # (c, P, K)
        t_reach = np.where(np.isnan(t_reach), np.inf, t_reach)
        slack = t_reach - taus[None, None, :]                                       # <= 0: there
        sh = np.min(slack[:, home, :], axis=1)                                      # (c, K)
        sa = np.min(slack[:, away, :], axis=1)
        reach = (sh <= 0) | (sa <= 0)
        k = np.where(reach.any(axis=1), np.argmax(reach, axis=1), len(taus) - 1)
        rows = np.arange(b - a)
        h_at, a_at = sh[rows, k], sa[rows, k]
        found = reach.any(axis=1)
        out["int_team"][a:b] = np.where(found, np.where(h_at <= a_at, 1.0, -1.0), 0.0)
        out["int_time_s"][a:b] = np.where(found, taus[k], INT_HORIZON + INT_STEP)
        out["int_margin_s"][a:b] = np.clip(a_at - h_at, -INT_HORIZON, INT_HORIZON)
        th, ta = np.min(t_reach[:, home, 0], axis=1), np.min(t_reach[:, away, 0], axis=1)
        out["int_now_diff_s"][a:b] = np.clip(np.where(np.isfinite(th) & np.isfinite(ta),
                                                      ta - th, np.nan), -5, 5)
    unknown = np.isnan(ball_i[:, 0])
    for v in out.values():
        v[unknown] = np.nan
    return out


def carrier(ball_i, bvel, players, pvel, team):
    dist = np.linalg.norm(players - ball_i[:, None, :], axis=2)                    # (N, P)
    bs = np.linalg.norm(bvel, axis=1)
    ps = np.linalg.norm(pvel, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(pvel * bvel[:, None, :], axis=2) / (ps * bs[:, None])
    both = (ps > MOVING) & (bs[:, None] > MOVING)
    align = np.where(both, np.nan_to_num(cos), 0.5)       # still ball or player: neutral
    inst = np.exp(-dist / CARRY_SCALE) * (0.5 + 0.5 * align)
    out = {}
    for w in CARRY_W:
        score = pd.DataFrame(inst).rolling(2 * w + 1, center=True, min_periods=1).mean().to_numpy()
        with np.errstate(all="ignore"):
            h = np.nanmax(np.where(team[None, :] == HOME, score, np.nan), axis=1)
            a = np.nanmax(np.where(team[None, :] == AWAY, score, np.nan), axis=1)
        out["carry_home_w%d" % w], out["carry_away_w%d" % w] = h, a
        out["carry_diff_w%d" % w] = h - a
    unknown = np.isnan(ball_i[:, 0])
    for v in out.values():
        v[unknown] = np.nan
    return out


def goalkeeper(ball_i, bspeed, players, team, hsign):
    px = players[:, :, 0]
    out = {}
    for side, k, s in (("home", HOME, hsign), ("away", AWAY, -hsign)):
        m = team == k
        x_own = np.where(s[:, None] > 0, px[:, m], PITCH_L - px[:, m])  # 0 = own goal line
        x_own = np.where(np.isnan(x_own), np.inf, x_own)
        deep = np.argmin(x_own, axis=1)
        gk = players[:, m][np.arange(len(players)), deep]
        d = np.linalg.norm(gk - ball_i, axis=1)
        d[~np.isfinite(np.min(x_own, axis=1))] = np.nan
        out["gk_dist_%s" % side] = d
        out["_hold_%s" % side] = ((d < GK_NEAR) & (bspeed < GK_SLOW)).astype(np.float32)
    out["gk_hold"] = out.pop("_hold_home") - out.pop("_hold_away")
    unknown = np.isnan(ball_i[:, 0])
    for v in out.values():
        v[unknown] = np.nan
    return out


def build_extra(match, ball=None, hsign=None):
    """DataFrame of the extra features, one row per frame of `match`."""
    from .features import home_attack_sign
    players, team = match["players"], match["team"]
    raw = match["ball"] if ball is None else ball
    if hsign is None:
        half = match["half"]
        kick = {h: int(k) for h, k in enumerate(match["kickoff"], start=1)} \
            if "kickoff" in match else None
        sign = home_attack_sign(players, team, kick)
        hsign = np.ones(len(players), np.float32)
        for h, sg in sign.items():
            hsign[half == h] = sg
    _, ball_i, _ = clean_ball(raw)
    bvel = _velocity(ball_i)
    pvel = _velocity(players)
    bspeed = np.linalg.norm(np.nan_to_num(bvel), axis=1)
    f = {}
    f.update(interception(ball_i, bvel, players, pvel, team))
    f.update(carrier(ball_i, bvel, players, pvel, team))
    f.update(goalkeeper(ball_i, bspeed, players, team, hsign))
    df = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})
    for col in ("int_team", "int_margin_s", "carry_diff_w12", "gk_hold"):
        for w in ROLL:
            df["%s_r%d" % (col, w)] = _rolling(df[col], w).astype(np.float32)
    return df
