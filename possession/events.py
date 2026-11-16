"""Rule-based detectors for the moments that open and close a dead ball.

Each detector returns the frames where its event happens. They have a handful of settings each,
so they can be fitted on the few dozen stoppages we know and still hold on the rest of the match:
the physics of a ball going out or being kicked off does not change from one minute to the next.

  ball_out      the ball crosses a line, or disappears next to a line while heading out
  slowdown      most players drop to walking at the same moment (whistle: foul, offside, ...)
  restart       a ball held still is suddenly struck, or reappears next to a line heading in
"""

import numpy as np
import pandas as pd

from .data import FPS, PITCH_L, PITCH_W
from .features import clean_ball

DEFAULTS = dict(
    OUT_MARGIN=0.5,        # m beyond the line before the ball counts as out
    OUT_RUN=3,             # frames it must stay out
    LINE_NEAR=2.5,         # m: "next to a line"
    VANISH=25,             # frames the ball must stay missing after vanishing near a line
    OUTWARD=2.0,           # m/s towards the line, for a vanish to count as going out
    WALK_ON=0.70,          # share of players walking that marks a collective slowdown
    WALK_HOLD=50,          # frames the slowdown must last
    STILL_SPEED=1.5,       # m/s: ball held or placed
    STILL_HOLD=25,         # frames it must stay still before the kick
    KICK_SPEED=6.0,        # m/s: the restart kick
    KICK_WITHIN=12,        # frames between the end of stillness and reaching KICK_SPEED
    REAPPEAR_GAP=25,       # frames missing before a reappearance can be a restart
    REAPPEAR_LINE=5.0,     # m: a reappearance must be this close to a line (0 = anywhere)
    WALK_OFF=0.45,         # share of players walking below which play has resumed
    SPEEDUP_AFTER=50,      # frames of collective walking needed before a speed-up counts
    MIN_SEP=50,            # frames: at most one event of a kind within this window
    SLOW_BALL=6.0,         # m/s: with > 0, a slowdown also needs a slow or missing ball
)


def _runs(mask):
    """[(start, stop)) runs of True."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _thin(frames, sep):
    out = []
    for f in sorted(frames):
        if not out or f - out[-1] >= sep:
            out.append(int(f))
    return np.array(out, int)


def _line_distance(x, y):
    return np.minimum.reduce([x, PITCH_L - x, y, PITCH_W - y])


def ball_out(raw_ball, p=DEFAULTS):
    _, ball_i, seen = clean_ball(raw_ball)
    x, y = ball_i[:, 0], ball_i[:, 1]
    m = p["OUT_MARGIN"]
    with np.errstate(invalid="ignore"):
        out = (x < -m) | (x > PITCH_L + m) | (y < -m) | (y > PITCH_W + m)
    events = [a for a, b in _runs(out) if b - a >= p["OUT_RUN"]]

    # vanishing next to a line while moving towards it
    vel = np.gradient(ball_i, axis=0) * FPS
    dist = _line_distance(x, y)
    # outward speed: velocity component pointing to the nearest line
    nx = np.select([x <= np.minimum.reduce([PITCH_L - x, y, PITCH_W - y]),
                    PITCH_L - x <= np.minimum(y, PITCH_W - y)], [-1.0, 1.0], 0.0)
    ny = np.where(nx == 0, np.where(y < PITCH_W / 2, -1.0, 1.0), 0.0)
    outward = vel[:, 0] * nx + vel[:, 1] * ny
    for a, b in _runs(~seen):
        if a == 0 or b - a < p["VANISH"]:
            continue
        last = a - 1
        if dist[last] <= p["LINE_NEAR"] and outward[last] >= p["OUTWARD"]:
            events.append(last + int(round(dist[last] / outward[last] * FPS)))
    return _thin(events, p["MIN_SEP"])


def walking_share(player_speed):
    """Share of tracked players below 2 m/s, smoothed over one second."""
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            walking = np.nanmean(np.where(np.isnan(player_speed), np.nan,
                                          player_speed < 2.0), axis=1)
    return pd.Series(walking).rolling(25, center=True, min_periods=1).mean().to_numpy()


def slowdown(player_speed, p=DEFAULTS, raw_ball=None):
    """Collective slowdown. With `raw_ball` and SLOW_BALL > 0, the ball must also be slow or
    missing during the first two seconds, which separates a stoppage from slow live play."""
    walk = walking_share(player_speed)
    events = [a for a, b in _runs(walk >= p["WALK_ON"]) if b - a >= p["WALK_HOLD"]]
    if raw_ball is not None and p.get("SLOW_BALL", 0) > 0:
        _, ball_i, seen = clean_ball(raw_ball)
        speed = np.linalg.norm(np.gradient(ball_i, axis=0) * FPS, axis=1)
        keep = []
        for a in events:
            w = slice(a, a + 2 * FPS)
            missing = np.mean(~seen[w])
            with np.errstate(invalid="ignore"):
                slow = np.nanmedian(speed[w]) if np.any(seen[w]) else 0.0
            if missing >= 0.5 or slow < p["SLOW_BALL"]:
                keep.append(a)
        events = keep
    return _thin(events, p["MIN_SEP"])


def speedup(player_speed, p=DEFAULTS):
    """End of a collective slowdown: players stop walking together (play has resumed)."""
    walk = walking_share(player_speed)
    events = []
    for a, b in _runs(walk >= p["WALK_ON"]):
        if b - a < p["SPEEDUP_AFTER"]:
            continue
        with np.errstate(invalid="ignore"):
            after = np.flatnonzero(walk[b:b + 10 * FPS] <= p["WALK_OFF"])
        if len(after):
            events.append(b + after[0])
    return _thin(events, p["MIN_SEP"])


def restart(raw_ball, p=DEFAULTS):
    _, ball_i, seen = clean_ball(raw_ball)
    speed = np.linalg.norm(np.gradient(ball_i, axis=0) * FPS, axis=1)
    speed = pd.Series(speed).rolling(3, center=True, min_periods=1).median().to_numpy()
    events = []
    with np.errstate(invalid="ignore"):
        still = speed < p["STILL_SPEED"]
    for a, b in _runs(still):
        if b - a < p["STILL_HOLD"]:
            continue
        window = speed[b:b + p["KICK_WITHIN"]]
        with np.errstate(invalid="ignore"):
            hit = np.flatnonzero(window >= p["KICK_SPEED"])
        if len(hit):
            events.append(b + hit[0] - 1)       # the kick happens just before the speed shows
    # throw-ins and goal kicks: ball reappears next to a line after being missing
    x, y = ball_i[:, 0], ball_i[:, 1]
    dist = _line_distance(x, y)
    for a, b in _runs(~seen):
        if b - a >= p["REAPPEAR_GAP"] and b < len(seen) and (
                p["REAPPEAR_LINE"] <= 0 or dist[b] <= p["REAPPEAR_LINE"]):
            events.append(b)
    return _thin(events, p["MIN_SEP"])
