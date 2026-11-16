"""Per-frame features for the possession classifier.

Three groups, all computed for every video frame:
  ball      cleaned ball track, gaps, speed, position relative to the lines, nearest players
  shape     team shape and movement, available even when the ball is not
  tags      Smart Tagging phases as a signed "which team is attacking" signal, and set pieces
plus centred rolling means of the most useful ones, so the classifier sees a few seconds of context.

Sign convention for team-level features: positive favours the home side (H).
"""

import numpy as np
import pandas as pd

from .data import (AWAY, FPS, HOME, N_FRAMES, PITCH_L, PITCH_W, halves, half_index,
                   load_tags, load_tracking)

# Params
MAX_BALL_SPEED = 45.0     # m/s; a jump faster than this on both sides of a point is a false detection
MAX_OUTSIDE = 8.0         # m; ball detections further outside the pitch than this are discarded
INTERP_MAX_GAP = 25       # frames; shorter ball gaps are filled by straight lines
GAP_CAP = 10 * FPS        # frames; gap lengths are capped here
NEAR = 3.0                # m; "close to the ball"
WALK = 2.0                # m/s; below this a player is walking
SPRINT = 5.5              # m/s
ROLL = (12, 50)           # half-widths, in frames, of the centred rolling means
TOUCH_DV = 4.0            # m/s: a change of ball velocity this large next to a player is a touch
TOUCH_NEAR = 2.5          # m: how close that player must be
TOUCH_GAP = 5             # frames: at most one touch within this many frames
TOUCH_CAP = 10 * FPS      # frames: "time since / until a touch" is capped here
CTL_CAP = 3 * FPS         # frames: control run lengths are capped here
CHASE_RADIUS = 15.0       # m: players this close to the ball count for "chasing"
PRESS_RADIUS = 5.0        # m: players this close to the ball count as pressure

ATTACK_CODES = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL"}
DEFEND_CODES = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}


def home_attack_sign(players, team, kickoffs=None):
    """{half: +1 if the home team attacks towards x = PITCH_L, else -1}, read at each kickoff,
    when both teams are necessarily in their own half. `kickoffs` {half: frame}, default ours."""
    if kickoffs is None:
        kickoffs = {half: (first if half == 1 else 113742) for half, first, _ in halves()}
    out = {}
    for half, kickoff in kickoffs.items():
        window = players[kickoff:kickoff + FPS]
        home_x = np.nanmean(window[:, team == HOME, 0])
        away_x = np.nanmean(window[:, team == AWAY, 0])
        out[half] = 1.0 if home_x < away_x else -1.0
    return out


def _gap_lengths(seen):
    """(frames since last seen, frames until next seen, length of the current gap); 0 when seen."""
    n = len(seen)
    idx = np.arange(n)
    last = np.where(seen, idx, -10**9)
    last = np.maximum.accumulate(last)
    nxt = np.where(seen, idx, 10**9)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    since = np.minimum(idx - last, GAP_CAP)
    until = np.minimum(nxt - idx, GAP_CAP)
    length = np.where(seen, 0, np.minimum(since + until - 1, GAP_CAP))
    return since.astype(np.float32), until.astype(np.float32), length.astype(np.float32)


def clean_ball(ball):
    """(cleaned raw ball, interpolated ball, seen mask). Drops far-outside points and isolated
    jumps, then fills short gaps linearly."""
    b = ball.copy()
    x, y = b[:, 0], b[:, 1]
    far = (x < -MAX_OUTSIDE) | (x > PITCH_L + MAX_OUTSIDE) | (y < -MAX_OUTSIDE) | \
          (y > PITCH_W + MAX_OUTSIDE)
    b[far] = np.nan

    seen = ~np.isnan(b[:, 0])
    idx = np.flatnonzero(seen)
    if len(idx) > 2:
        pts = b[idx]
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.diff(idx) * FPS
        jump_in = np.r_[np.inf, step]
        jump_out = np.r_[step, np.inf]
        bad = (jump_in > MAX_BALL_SPEED) & (jump_out > MAX_BALL_SPEED)
        b[idx[bad]] = np.nan
    seen = ~np.isnan(b[:, 0])

    interp = b.copy()
    since, until, length = _gap_lengths(seen)
    fill = (~seen) & (length <= INTERP_MAX_GAP) & (since < GAP_CAP) & (until < GAP_CAP)
    idx = np.flatnonzero(seen)
    for k in range(2):
        interp[fill, k] = np.interp(np.flatnonzero(fill), idx, b[idx, k])
    return b, interp, seen


def detect_touches(ball_i, players, team):
    """(N,) int8 per frame: HOME or AWAY where a player of that team touches the ball, else -1.

    A touch is a sudden change of the ball's velocity (over +-3 frames, after light smoothing)
    with a player within TOUCH_NEAR metres; the strongest change wins within TOUCH_GAP frames.
    """
    n = len(ball_i)
    pos = pd.DataFrame(ball_i).rolling(5, center=True, min_periods=3).mean().to_numpy()
    vel = np.full_like(pos, np.nan)
    vel[2:-2] = (pos[4:] - pos[:-4]) * FPS / 4
    dv = np.full(n, np.nan)
    dv[3:-3] = np.linalg.norm(vel[6:] - vel[:-6], axis=1)
    with np.errstate(invalid="ignore"):
        cand = np.flatnonzero(dv > TOUCH_DV)
    dist = np.linalg.norm(players - ball_i[:, None, :], axis=2)
    out = np.full(n, -1, np.int8)
    taken = np.zeros(n, bool)
    for t in cand[np.argsort(-dv[cand], kind="stable")]:
        if taken[max(0, t - TOUCH_GAP):t + TOUCH_GAP + 1].any():
            continue
        d = dist[t]
        if np.all(np.isnan(d)):
            continue
        k = int(np.nanargmin(d))
        if d[k] <= TOUCH_NEAR:
            out[t] = team[k]
            taken[t] = True
    return out


def _run_lengths(values):
    """(frames since the value last changed, frames until it next changes), capped at CTL_CAP.
    NaN values break runs."""
    n = len(values)
    back = np.zeros(n)
    fwd = np.zeros(n)
    for i in range(1, n):
        same = values[i] == values[i - 1] and not np.isnan(values[i])
        back[i] = back[i - 1] + 1 if same else 0
    for i in range(n - 2, -1, -1):
        same = values[i] == values[i + 1] and not np.isnan(values[i])
        fwd[i] = fwd[i + 1] + 1 if same else 0
    return np.minimum(back, CTL_CAP), np.minimum(fwd, CTL_CAP)


def control_features(ball_i, players, team):
    """Who controls the ball over time, not just who is closest on one frame.

    ctl_slot_run / ctl_team_run   how long the same player / the same team has been nearest to
                                  the ball (signed: + home, - away), and _ahead: how long it stays so
    ctl_chase_home/away/diff      mean speed TOWARDS the ball of each team's players within
                                  CHASE_RADIUS. Measured: the team moving TOWARDS the ball is
                                  usually the one that has it (right ~79% over +-2 s), not the
                                  defending team
    ctl_press_home/away           players of each team within PRESS_RADIUS of the ball
    """
    n = len(ball_i)
    dist = np.linalg.norm(players - ball_i[:, None, :], axis=2)
    known = ~np.all(np.isnan(dist), axis=1)
    nearest = np.full(n, np.nan)
    nearest[known] = np.nanargmin(np.where(np.isnan(dist[known]), np.inf, dist[known]), axis=1)
    near_team = np.full(n, np.nan)
    near_team[known] = team[nearest[known].astype(int)]
    sign = np.where(near_team == HOME, 1.0, np.where(near_team == AWAY, -1.0, np.nan))
    f = {}
    back, fwd = _run_lengths(nearest)
    f["ctl_slot_run"] = back / FPS * sign
    f["ctl_slot_ahead"] = fwd / FPS * sign
    back, fwd = _run_lengths(near_team)
    f["ctl_team_run"] = back / FPS * sign
    f["ctl_team_ahead"] = fwd / FPS * sign

    pos = pd.DataFrame(players.reshape(n, -1)).rolling(5, center=True, min_periods=3).mean()
    vel = np.gradient(pos.to_numpy().reshape(players.shape), axis=0) * FPS
    to_ball = ball_i[:, None, :] - players
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = to_ball / np.linalg.norm(to_ball, axis=2, keepdims=True)
        towards = np.sum(vel * unit, axis=2)                       # m/s towards the ball
        close = dist <= CHASE_RADIUS
        for side, k in (("home", HOME), ("away", AWAY)):
            m = (team == k)[None, :] & close
            f["ctl_chase_%s" % side] = np.nansum(np.where(m, towards, 0), axis=1) / np.maximum(
                m.sum(axis=1), 1)
            f["ctl_press_%s" % side] = np.sum((team == k)[None, :] & (dist <= PRESS_RADIUS),
                                              axis=1).astype(float)
    f["ctl_chase_diff"] = f["ctl_chase_away"] - f["ctl_chase_home"]   # > 0: away chases more
    for key in ("ctl_chase_home", "ctl_chase_away", "ctl_chase_diff", "ctl_press_home",
                "ctl_press_away"):
        f[key][~known] = np.nan
    for w in ROLL:
        f["ctl_chase_diff_r%d" % w] = pd.Series(f["ctl_chase_diff"]).rolling(
            2 * w + 1, center=True, min_periods=1).mean().to_numpy()
    return f


def touch_features(touch):
    """Signals from the touch sequence. Sign: +1 home, -1 away."""
    n = len(touch)
    idx = np.arange(n)
    has = touch >= 0
    sign = np.where(touch == HOME, 1.0, np.where(touch == AWAY, -1.0, 0.0))
    last = np.maximum.accumulate(np.where(has, idx, -1))
    nxt = np.minimum.accumulate(np.where(has, idx, n)[::-1])[::-1]
    f = {}
    f["touch_last"] = np.where(last >= 0, sign[np.maximum(last, 0)], 0.0)
    f["touch_next"] = np.where(nxt < n, sign[np.minimum(nxt, n - 1)], 0.0)
    f["touch_since_s"] = np.where(last >= 0, np.minimum(idx - last, TOUCH_CAP), TOUCH_CAP) / FPS
    f["touch_until_s"] = np.where(nxt < n, np.minimum(nxt - idx, TOUCH_CAP), TOUCH_CAP) / FPS
    # how many touches in a row the last-touching team has made: 2+ looks like control
    events = np.flatnonzero(has)
    streak_at = np.zeros(n)
    run = 0
    prev_team = -1
    prev_of = np.full(n, 0.0)          # team of the touch before the last one
    for e in events:
        run = run + 1 if touch[e] == prev_team else 1
        streak_at[e] = run
        prev_of[e] = 1.0 if prev_team == HOME else -1.0 if prev_team == AWAY else 0.0
        prev_team = touch[e]
    safe = np.maximum(last, 0)
    f["touch_streak"] = np.where(last >= 0, np.minimum(streak_at[safe], 5), 0.0)
    f["touch_changed"] = np.where(last >= 0, (prev_of[safe] != 0) & (prev_of[safe] != f["touch_last"]), 0.0) * 1.0
    for w in ROLL:
        f["touch_bal_r%d" % w] = pd.Series(sign).rolling(2 * w + 1, center=True,
                                                         min_periods=1).sum().to_numpy()
    return f


def event_features(raw_ball, player_speed):
    """Seconds since the last and until the next stoppage event of each kind (possession.events),
    capped at 10 s."""
    from . import events
    n = len(raw_ball)
    idx = np.arange(n)
    found = {
        "ev_out": events.ball_out(raw_ball),
        "ev_slow": events.slowdown(player_speed, raw_ball=raw_ball),
        "ev_kick": events.restart(raw_ball),
        "ev_speedup": events.speedup(player_speed),
    }
    f = {}
    for name, frames in found.items():
        mark = np.zeros(n, bool)
        mark[frames[(frames >= 0) & (frames < n)]] = True
        last = np.maximum.accumulate(np.where(mark, idx, -10**9))
        nxt = np.minimum.accumulate(np.where(mark, idx, 10**9)[::-1])[::-1]
        f[name + "_since_s"] = np.minimum(idx - last, TOUCH_CAP) / FPS
        f[name + "_until_s"] = np.minimum(nxt - idx, TOUCH_CAP) / FPS
    return f


OWN_R, OWN_K, OWN_DELAY = 2.0, 25, 12     # tools/ownership_rule.py, best balanced rule


def ownership_rule(d_home, d_away, half):
    """+1 / -1: team that "owns" the ball under a persistence rule -- it passes to the other team
    once one of its players has been within OWN_R m of the ball (and nearest) for OWN_K frames in
    a row, dated back to the start of that control and shifted by OWN_DELAY. 0 before any owner."""
    with np.errstate(invalid="ignore"):
        ev = np.where((d_home <= OWN_R) & (d_home < d_away), 1,
                      np.where((d_away <= OWN_R) & (d_away < d_home), -1, 0))
    out = np.zeros(len(ev))
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        seg = ev[idx]
        own = np.zeros(len(seg))
        cur, cand, streak = 0, 0, 0
        for i, v in enumerate(seg):
            if v != 0 and v != cur:
                streak = streak + 1 if v == cand else 1
                cand = v
                if streak >= OWN_K or cur == 0:
                    start = max(0, i - streak + 1 + (OWN_DELAY if cur != 0 else 0))
                    own[start:] = v
                    cur, streak = v, 0
            elif v == cur:
                streak = 0
        out[idx] = own
    return out


def _rolling(series, half_width):
    return series.rolling(2 * half_width + 1, center=True, min_periods=1).mean()


def ball_features(raw_ball, players, team, hsign, prefix=""):
    """Ball signals from one ball track (metres, NaN = not seen), keys prefixed with `prefix`."""
    f = {}
    ball, ball_i, seen = clean_ball(raw_ball)
    since, until, gap = _gap_lengths(seen)
    f["ball_seen"] = seen.astype(np.float32)
    f["ball_raw_seen"] = (~np.isnan(raw_ball[:, 0])).astype(np.float32)
    f["ball_interp"] = ((~seen) & ~np.isnan(ball_i[:, 0])).astype(np.float32)
    f["gap_since_s"] = since / FPS
    f["gap_until_s"] = until / FPS
    f["gap_len_s"] = gap / FPS

    bx, by = ball_i[:, 0], ball_i[:, 1]
    # x measured towards the goal the home team attacks: 0 = home goal line, L = away goal line
    bx_home = np.where(hsign > 0, bx, PITCH_L - bx)
    f["ball_x_home"] = bx_home
    f["ball_y_abs"] = np.abs(by - PITCH_W / 2)
    f["ball_to_touchline"] = np.minimum(by, PITCH_W - by)
    f["ball_to_goalline"] = np.minimum(bx, PITCH_L - bx)
    f["ball_out"] = ((bx < 0) | (bx > PITCH_L) | (by < 0) | (by > PITCH_W)).astype(np.float32)
    f["ball_out"][np.isnan(bx)] = np.nan

    vel = np.gradient(ball_i, axis=0) * FPS
    bspeed = np.linalg.norm(vel, axis=1)
    f["ball_speed"] = pd.Series(bspeed).rolling(5, center=True, min_periods=1).median().to_numpy()
    f["ball_accel"] = np.abs(np.gradient(f["ball_speed"])) * FPS

    dist = np.linalg.norm(players - ball_i[:, None, :], axis=2)  # (N, S), NaN where unknown
    for side, k in (("home", HOME), ("away", AWAY)):
        d = dist[:, team == k]
        with np.errstate(all="ignore"):
            f["d_%s" % side] = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
        f["d_%s" % side][~np.isfinite(f["d_%s" % side])] = np.nan
        f["n_%s_near" % side] = np.sum(d < NEAR, axis=1).astype(np.float32)
        f["n_%s_near" % side][np.isnan(bx)] = np.nan
    f["d_diff"] = f["d_away"] - f["d_home"]          # > 0: a home player is closer
    f["nearest_is_home"] = np.where(np.isnan(f["d_diff"]), np.nan, (f["d_diff"] > 0) * 1.0)
    f.update(touch_features(detect_touches(ball_i, players, team)))
    f.update(control_features(ball_i, players, team))
    return {prefix + k: v for k, v in f.items()}


def build(ball=None, vision_ball=None, refresh_tracking=False, match=None):
    """DataFrame of features, one row per video frame.

    `ball` replaces the tracked ball positions (metres, NaN = not seen); used to rebuild the
    features with the ball artificially removed, for training copies and stress tests.
    `vision_ball`, if given, is a second ball track (the tracked ball with its gaps filled from
    video); its signals are added with a "vis_" prefix, next to the tracking-based ones, so the
    model sees both "the tracking lost the ball" and where vision thinks the ball is.
    `match`, if given, is another match in the same format as load_tracking plus "half" (per
    frame) and "kickoff" (first frame of each half), e.g. possession/dfl.py; it has no Smart
    Tagging, so the tag signals are zero.
    """
    if match is None:
        t = load_tracking(refresh_tracking)
        half, kickoffs, tags = half_index(), None, load_tags()
    else:
        t = match
        half = match["half"]
        kickoffs = {h: int(k) for h, k in enumerate(match["kickoff"], start=1)}
        tags = None
    n_frames = len(t["players"])
    raw_ball = t["ball"] if ball is None else ball
    players, speed, team = t["players"], t["player_speed"], t["team"]
    sign_by_half = home_attack_sign(players, team, kickoffs)
    hsign = np.zeros(n_frames, np.float32)
    for h, s in sign_by_half.items():
        hsign[half == h] = s
    hsign[half == 0] = 1.0

    f = ball_features(raw_ball, players, team, hsign)
    f.update(event_features(raw_ball, speed))
    f["own_rule"] = ownership_rule(f["d_home"], f["d_away"], half)
    if vision_ball is not None:
        vis = ball_features(vision_ball, players, team, hsign, prefix="vis_")
        f.update({k: v for k, v in vis.items() if not k.startswith(("vis_gap", "vis_ball_raw"))})

    # -- team shape and movement (no ball needed) ---------------------------------------------
    px = players[:, :, 0]
    py = players[:, :, 1]
    with np.errstate(all="ignore"):
        for side, k, s in (("home", HOME, hsign), ("away", AWAY, -hsign)):
            m = team == k
            x_att = np.where(s[:, None] > 0, px[:, m], PITCH_L - px[:, m])  # towards own attack
            f["%s_n" % side] = np.sum(~np.isnan(x_att), axis=1).astype(np.float32)
            f["%s_adv" % side] = np.nanmean(x_att, axis=1)
            f["%s_back" % side] = np.nanpercentile(x_att, 15, axis=1)
            f["%s_len" % side] = np.nanstd(x_att, axis=1)
            f["%s_wid" % side] = np.nanstd(py[:, m], axis=1)
            sp = speed[:, m]
            f["%s_speed" % side] = np.nanmean(sp, axis=1)
            f["%s_walk" % side] = np.nanmean(np.where(np.isnan(sp), np.nan, sp < WALK), axis=1)
            f["%s_sprint" % side] = np.nanmean(np.where(np.isnan(sp), np.nan, sp > SPRINT),
                                               axis=1)
        f["adv_diff"] = f["home_adv"] - f["away_adv"]
        f["back_diff"] = f["home_back"] - f["away_back"]
        f["len_diff"] = f["home_len"] - f["away_len"]
        f["wid_diff"] = f["home_wid"] - f["away_wid"]
        f["speed_all"] = np.nanmean(speed, axis=1)
        f["walk_all"] = np.nanmean(np.where(np.isnan(speed), np.nan, speed < WALK), axis=1)
        # team centroid motion towards the home team's attack, from positions (m/s)
        cx = np.nanmean(np.where(hsign[:, None] > 0, px, PITCH_L - px), axis=1)
        f["block_drift"] = np.gradient(pd.Series(cx).rolling(25, center=True,
                                                              min_periods=1).mean().to_numpy()) * FPS

    # -- Smart Tagging ------------------------------------------------------------------------
    attack = np.zeros(n_frames, np.float32)
    setpiece = np.zeros(n_frames, np.float32)
    setpiece_team = np.zeros(n_frames, np.float32)
    sp_starts = []
    for row in (tags.itertuples() if tags is not None else ()):
        a, b = max(0, row.start), min(n_frames, row.end)
        team_sign = 1.0 if row.team == "H" else -1.0 if row.team == "A" else 0.0
        if row.code in ATTACK_CODES:
            attack[a:b] += team_sign
        elif row.code in DEFEND_CODES:
            attack[a:b] -= team_sign        # the defending team does not have the ball
        elif row.code == "SET PIECES":
            setpiece[a:b] = 1.0
            setpiece_team[a:b] += team_sign
            sp_starts.append(a)
    f["tag_attack"] = attack
    f["tag_setpiece"] = setpiece
    f["tag_setpiece_team"] = setpiece_team
    starts = np.zeros(n_frames, bool)
    starts[sp_starts] = True
    s_since, s_until, _ = _gap_lengths(starts)
    f["setpiece_since_s"] = s_since / FPS
    f["setpiece_until_s"] = s_until / FPS

    df = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})

    # -- context ------------------------------------------------------------------------------
    for col in ("ball_seen", "d_diff", "nearest_is_home", "ball_speed", "speed_all", "walk_all",
                "adv_diff", "block_drift", "tag_attack", "n_home_near", "n_away_near",
                "ball_out", "vis_d_diff", "vis_nearest_is_home", "vis_ball_speed",
                "vis_n_home_near", "vis_n_away_near", "vis_ball_out"):
        if col not in df:
            continue
        for w in ROLL:
            df["%s_r%d" % (col, w)] = _rolling(df[col], w).astype(np.float32)

    df["half"] = half
    return df, sign_by_half
