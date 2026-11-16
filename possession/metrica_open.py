"""External training data from the challenge's own provider: Metrica Sports' open sample matches
(github.com/metrica-sports/sample-data, tools/download_metrica.sh -> external/metrica/).

Games 1 and 2: 25 Hz tracking of both teams and the ball (CSV, 0-1 coordinates), synchronised
event data. Metrica's events carry its own notion of possession: RECOVERY marks the moment a team
gains the ball, BALL LOST the moment it gives it away, so possession labels follow Metrica's
convention much more closely than DFL's per-touch flag (about 5 changes per live minute, like our
ground truth, against 7-11 for DFL).

Labels per frame (0 H, 1 A, 2 D, 3 X), from the events:
  possession  the team of the last controlled action (PASS, RECOVERY, SET PIECE, SHOT, BALL LOST);
              it changes only when the other team recovers or plays the ball
  dead ball   from the end of a BALL OUT, a shot that goes out or in, or a foul (FAULT
              RECEIVED) until the restart: the next play of the ball (SET PIECE, PASS, SHOT, BALL
              LOST; a RECOVERY in between, the keeper collecting the ball for a goal kick, gives
              the team but does not end it). A SET PIECE after anything else (offside, ball lost
              out of play without a BALL OUT event, ~1 restart in 5) is dead from the end of the
              event before it, as in possession/pff.py
`load(game)` returns the same dict as possession/dfl.load.
"""

import os

import numpy as np
import pandas as pd

from .data import CACHE, FPS, PITCH_L, PITCH_W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "external", "metrica")
GAMES = (1, 2)
CONTROL = ("PASS", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST")
GAP = 250


def _tracking(game, side):
    path = os.path.join(DIR, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(game, side))
    df = pd.read_csv(path, skiprows=2)
    cols = list(df.columns)
    players = []
    for i, c in enumerate(cols):
        if c.startswith("Player"):
            players.append((cols[i], cols[i + 1]))
    ball = ("Ball", cols[cols.index("Ball") + 1])
    return df, players, ball


def _labels(events, n):
    """Per raw frame: 0 H, 1 A, 2 D, 3 X (before the first event)."""
    team_code = {"Home": 0, "Away": 1}
    owner = np.full(n, -1, np.int64)
    dead = np.zeros(n, bool)
    ev = events.sort_values("start", kind="stable")
    cur, last, dead_from, prev_end = -1, 0, None, None
    for team, typ, sub, f, f_end in zip(ev.team, ev.type, ev.subtype.astype(str), ev.start,
                                        ev.end):
        if f < 0 or f >= n:
            continue
        owner[last:f] = cur
        last = f
        if typ in CONTROL and team in team_code:
            if typ == "SET PIECE" and dead_from is None:
                dead_from = prev_end
            if dead_from is not None and typ != "RECOVERY":      # the restart: first play
                dead[dead_from:f] = True
                dead_from = None
            cur = team_code[team]
        if typ == "BALL OUT" or (typ == "SHOT" and ("GOAL" in sub or "OUT" in sub)):
            dead_from = f_end if f_end > 0 else f
        elif typ == "FAULT RECEIVED":
            dead_from = f
        if typ not in ("CARD", "CHALLENGE"):
            prev_end = max(f_end, f) if f_end > 0 else f
    owner[last:] = cur
    lab = np.where(owner >= 0, owner, 3).astype(np.int8)
    lab[dead] = 2
    return lab


CONTROL_RULES = {
    # the challenge's definition (CHALLENGE.md): a change of possession needs CONTROLLED
    # possession; a touch without control, a deflection or an aerial duel won without keeping the
    # ball do not count, and while the ball is contested or loose possession stays with the last
    # team that had it. A spell won in open play counts only if its team shows control:
    "lite": dict(seconds=2.0, passes=1),     # a pass or a shot, or the ball kept 2 s
    "strict": dict(seconds=3.0, passes=3),   # the definition's own proxies: a few seconds, or
                                             # more than two consecutive passes (or a shot)
}


def _control(lab, events, rule):
    """Spells won in open play without control (CONTROL_RULES[rule]) given back to the team
    that had the ball before; spells starting at a restart are always kept."""
    seconds, passes = CONTROL_RULES[rule]["seconds"], CONTROL_RULES[rule]["passes"]
    team_code = {"Home": 0, "Away": 1}
    ev = events[events.team.isin(team_code)]
    ev_frame, ev_team, ev_type = ev.start.to_numpy(), ev.team.map(team_code).to_numpy(), \
        ev.type.to_numpy()
    lab = lab.copy()
    live = np.isin(lab, (0, 1))
    edges = np.flatnonzero(np.diff(np.r_[-1, lab, -1]) != 0)
    for s, e in zip(edges[:-1], edges[1:]):
        k = lab[s]
        if k not in (0, 1) or s == 0 or not live[s - 1] or lab[s - 1] == k:
            continue                                  # a restart, or merged into its neighbour
        mine = (ev_frame >= s) & (ev_frame < e) & (ev_team == k)
        n_pass = np.sum(mine & (ev_type == "PASS"))
        shot = np.any(mine & (ev_type == "SHOT"))
        if (e - s) >= seconds * FPS or n_pass >= passes or shot:
            continue
        lab[s:e] = lab[s - 1]
    return lab


def labels(game_id, rule):
    """Laid-out labels of Metrica game "M1".."M3" under CONTROL_RULES[rule] (or "events": the
    default labels of load()), cached as cache/metrica_<n>_<rule>.npy."""
    if rule == "events":
        m = load_game3() if game_id == "M3" else load(int(game_id[1:]))
        return m["labels"]
    path = os.path.join(CACHE, "metrica_{}_{}.npy".format(game_id[1:], rule))
    if os.path.exists(path):
        return np.load(path)
    lab = (load_game3(rule=rule) if game_id == "M3" else load(int(game_id[1:]), rule=rule))[
        "labels"]
    np.save(path, lab)
    return lab


def load(game, refresh=False, rule=None):
    path = os.path.join(CACHE, "metrica_{}.npz".format(game))
    if os.path.exists(path) and not refresh and rule is None:
        return dict(np.load(path))
    home, hp, hb = _tracking(game, "Home")
    away, ap, _ = _tracking(game, "Away")
    events = pd.read_csv(os.path.join(DIR, "Sample_Game_{}_RawEventsData.csv".format(game)))
    events.columns = ["team", "type", "subtype", "period", "start", "start_s", "end", "end_s",
                      "from", "to", "sx", "sy", "ex", "ey"]
    frame = home["Frame"].to_numpy()
    period = home["Period"].to_numpy()
    n_raw = int(frame.max()) + 1
    P = np.full((n_raw, len(hp) + len(ap), 2), np.nan, np.float32)
    for j, (cx, cy) in enumerate(hp):
        P[frame, j] = np.column_stack([home[cx], home[cy]])
    aframe = away["Frame"].to_numpy()
    for j, (cx, cy) in enumerate(ap):
        P[aframe, len(hp) + j] = np.column_stack([away[cx], away[cy]])
    B = np.full((n_raw, 2), np.nan, np.float32)
    B[frame] = np.column_stack([home[hb[0]], home[hb[1]]])
    P *= np.array([PITCH_L, PITCH_W], np.float32)
    B *= np.array([PITCH_L, PITCH_W], np.float32)
    per = np.zeros(n_raw, np.int8)
    per[frame] = period
    lab_raw = _labels(events, n_raw)
    if rule is not None:
        lab_raw = _control(lab_raw, events, rule)
    out = _layout(P, B, per, lab_raw, np.r_[np.zeros(len(hp)), np.ones(len(ap))])
    if rule is None:                          # control-rule variants: labels() caches them
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, **out)
    return out


def _layout(P, B, per, lab_raw, team):
    """Lay the two periods end to end with a gap, like possession/dfl.py; same dict as dfl.load."""
    pieces, kick, half_parts, offset = [], [], [], 0
    for h in (1, 2):
        idx = np.flatnonzero(per == h)
        if not len(idx):
            continue
        pieces.append(idx)
        kick.append(offset)
        offset += len(idx) + GAP
    total = offset - GAP
    players = np.full((total, P.shape[1], 2), np.nan, np.float32)
    ball = np.full((total, 2), np.nan, np.float32)
    labels = np.full(total, 3, np.int8)
    half = np.zeros(total, np.int8)
    for h, (idx, k) in enumerate(zip(pieces, kick), start=1):
        players[k:k + len(idx)] = P[idx]
        ball[k:k + len(idx)] = B[idx]
        labels[k:k + len(idx)] = lab_raw[idx]
        half[k:k + len(idx)] = h
    speed = np.full(players.shape[:2], np.nan, np.float32)
    speed[1:] = np.linalg.norm(np.diff(players, axis=0), axis=2) * FPS
    speed = pd.DataFrame(speed).rolling(5, center=True, min_periods=1).mean().to_numpy(np.float32)
    bspeed = np.full(total, np.nan, np.float32)
    bspeed[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
    out = {"players": players, "player_speed": speed, "ball": ball,
           "ball_z": np.zeros(total, np.float32), "ball_speed": bspeed,
           "team": np.asarray(team).astype(np.int8),
           "labels": labels, "half": half, "kickoff": np.array(kick)}
    return out


def _epts_specs(meta_path):
    """(team per player id, ordered player ids, [(first frame, last frame, [player ids in line
    order])]) from an EPTS metadata file: the order of the players in a tracking line changes
    with substitutions, one DataFormatSpecification per stretch of frames."""
    import xml.etree.ElementTree as ET
    root = ET.parse(meta_path).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    team_of = {p.get("id"): (0 if p.get("teamId") == teams[0] else 1) for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        order = [channel_player[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")
                 if r.get("playerChannelId", "").endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    return team_of, list(team_of), specs


def load_game3(refresh=False, rule=None):
    """Sample game 3 (EPTS tracking like the challenge's, JSON events), as load() returns."""
    path = os.path.join(CACHE, "metrica_3.npz")
    if os.path.exists(path) and not refresh and rule is None:
        return dict(np.load(path))
    import json
    team_of, ids, specs = _epts_specs(os.path.join(DIR, "Sample_Game_3_metadata.xml"))
    slot = {pid: k for k, pid in enumerate(ids)}
    lines = open(os.path.join(DIR, "Sample_Game_3_tracking.txt")).read().splitlines()
    n_raw = int(lines[-1].split(":", 1)[0]) + 1
    P = np.full((n_raw, len(ids), 2), np.nan, np.float32)
    B = np.full((n_raw, 2), np.nan, np.float32)
    k = 0
    for line in lines:
        f, players, ball = line.split(":")
        f = int(f)
        while not (specs[k][0] <= f <= specs[k][1]) and k + 1 < len(specs):
            k += 1
        xy = np.array(players.replace(";", ",").split(","), np.float32).reshape(-1, 2)
        P[f, [slot[pid] for pid in specs[k][2]]] = xy
        B[f] = np.array(ball.split(","), np.float32)[:2]
    P *= np.array([PITCH_L, PITCH_W], np.float32)
    B *= np.array([PITCH_L, PITCH_W], np.float32)
    ev = json.load(open(os.path.join(DIR, "Sample_Game_3_events.json")))["data"]
    teams = sorted({e["team"]["id"] for e in ev if e.get("team")})
    home_id = [t for t in teams if t.endswith("A")][0]
    rows = [("Home" if (e.get("team") or {}).get("id") == home_id else "Away", e["type"]["name"],
             (e.get("subtypes") or {}).get("name") if isinstance(e.get("subtypes"), dict)
             else "", e["period"], e["start"]["frame"], e["end"]["frame"]) for e in ev]
    events = pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start", "end"])
    lab_raw = _labels(events, n_raw)
    if rule is not None:
        lab_raw = _control(lab_raw, events, rule)
    # period of every tracked frame: before / after the first event of the second period
    second = int(events[events.period == 2].start.min())
    seen = ~np.isnan(P[:, :, 0]).all(axis=1)
    per = np.where(seen, np.where(np.arange(n_raw) >= second, 2, 1), 0).astype(np.int8)
    out = _layout(P, B, per, lab_raw, np.array([team_of[pid] for pid in ids]))
    if rule is None:                          # control-rule variants: labels() caches them
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, **out)
    return out
