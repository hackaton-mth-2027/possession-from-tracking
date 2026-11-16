"""External training data: PFF FC / Gradient Sports 2022 World Cup (64 matches, tools/download_pff.sh
-> external/pff/).

Broadcast tracking at 29.97 Hz: all 22 players (positions estimated by PFF while off camera) and
the ball in 3D, plus event data with the team of every on-the-ball action. Players: PFF's smoothed
positions; ball: the raw one (seen or estimated), whose gaps already follow the game state as ours
do (one match: missing on 5% of live frames, 66% of dead-ball frames; ours 11% / 72%), whereas the
smoothed ball is missing on every dead-ball frame. Resampled to our 25 Hz.

Labels per frame (0 H, 1 A, 2 D, 3 X), from the events, in Metrica's event-based convention
(possession/metrica_open.py) rather than DFL's per-touch flag:
  possession  the team of the last controlled on-the-ball action (pass, cross, shot, clearance,
              carry, reception, touch); challenges and rebounds do not change it
  dead ball   from a ball out, or from the end of the last action before a set-piece restart
              (throw-in, corner, goal kick, free kick, kick-off, penalty), until that restart
`load(game_id)` returns the same dict as possession/dfl.load; regulation time only (periods 1-2).
"""

import bz2
import glob
import json
import os

import numpy as np

from .data import CACHE, FPS, PITCH_L, PITCH_W
from .metrica_open import _layout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "external", "pff")
CONTROL = ("PA", "CR", "SH", "CL", "BC", "IT", "TC")     # on-the-ball actions that show control
RESTART = ("T", "C", "G", "F", "K", "P")                 # set-piece types other than open play


def games():
    """Ids of the matches whose tracking, events and metadata are all present."""
    ids = []
    for path in sorted(glob.glob(os.path.join(DIR, "Tracking Data", "*.jsonl.bz2"))):
        gid = os.path.basename(path).split(".")[0]
        if all(os.path.exists(os.path.join(DIR, d, gid + ".json")) for d in ("Event Data",
                                                                               "Metadata")):
            ids.append(gid)
    return ids


def _metadata(gid):
    with open(os.path.join(DIR, "Metadata", gid + ".json")) as fh:
        meta = json.load(fh)
    return meta[0] if isinstance(meta, list) else meta


def _labels(gid, times):
    """Label of each tracking frame (video times in s) from the match's events."""
    with open(os.path.join(DIR, "Event Data", gid + ".json")) as fh:
        events = sorted(json.load(fh), key=lambda e: (e["startTime"], e.get("sequence") or 0))
    owner = np.full(len(times), -1, np.int64)
    dead = np.zeros(len(times), bool)
    cur, last_t, dead_from, prev_end = -1, None, None, None
    for e in events:
        ge = e.get("gameEvents") or {}
        kind, t = ge.get("gameEventType"), e["startTime"]
        i = np.searchsorted(times, t)
        if last_t is not None:
            owner[np.searchsorted(times, last_t):i] = cur
        last_t = t
        if kind == "OUT":
            dead_from = t if dead_from is None else dead_from
        elif kind in ("OTB", "FIRSTKICKOFF", "SECONDKICKOFF") and ge.get("homeTeam") is not None:
            pe = (e.get("possessionEvents") or {}).get("possessionEventType")
            if ge.get("setpieceType") in RESTART and prev_end is not None:
                start = dead_from if dead_from is not None else prev_end
                dead[np.searchsorted(times, start):i] = True
            dead_from = None
            if pe in CONTROL or kind != "OTB":
                cur = 0 if ge["homeTeam"] else 1
            prev_end = e.get("endTime") or t
        elif kind == "END":
            cur = -1
    if last_t is not None:
        owner[np.searchsorted(times, last_t):] = cur
    lab = np.where(owner >= 0, owner, 3).astype(np.int8)
    lab[dead & (lab < 3)] = 2
    return lab


PASSING = ("PA", "CR", "SH")     # actions that show control: pass, cross, shot


def labels(gid, rule):
    """Laid-out labels of match `gid` under possession/metrica_open.CONTROL_RULES[rule]: a spell
    won in open play counts only if its team makes a pass, a cross or a shot, or keeps the ball
    long enough; otherwise it goes back to the team before. Cached as cache/pff_<gid>_<rule>.npy."""
    from .metrica_open import CONTROL_RULES
    gid = str(gid)
    path = os.path.join(CACHE, "pff_{}_{}.npy".format(gid, rule))
    if os.path.exists(path):
        return np.load(path)
    seconds, passes = CONTROL_RULES[rule]["seconds"], CONTROL_RULES[rule]["passes"]
    m = load(gid)
    lab = m["labels"].copy()
    meta = _metadata(gid)
    with open(os.path.join(DIR, "Event Data", gid + ".json")) as fh:
        events = json.load(fh)
    times = np.full(len(lab), np.nan)
    for h, k0 in enumerate(m["kickoff"], start=1):
        idx = np.flatnonzero(m["half"] == h)
        start = meta.get("startPeriod{}".format(h))
        if start is None:       # the period's first event: its kick-off, the first tracked frame
            start = min(e["startTime"] for e in events
                        if (e.get("gameEvents") or {}).get("period") == h)
        times[idx] = start + np.arange(len(idx)) / FPS
    ev_t = np.array([e["startTime"] for e in events])
    ev_team = np.array([{True: 0, False: 1}.get((e.get("gameEvents") or {}).get("homeTeam"), -1)
                        for e in events])
    ev_pass = np.array([(e.get("possessionEvents") or {}).get("possessionEventType") in PASSING
                        for e in events])
    live = np.isin(lab, (0, 1))
    edges = np.flatnonzero(np.diff(np.r_[-1, lab, -1]) != 0)
    for s, e in zip(edges[:-1], edges[1:]):
        k = lab[s]
        if k not in (0, 1) or s == 0 or not live[s - 1] or lab[s - 1] == k:
            continue
        mine = (ev_t >= times[s]) & (ev_t < times[e - 1] + 1.0 / FPS) & (ev_team == k)
        if (e - s) >= seconds * FPS or np.sum(mine & ev_pass) >= passes:
            continue
        lab[s:e] = lab[s - 1]
    np.save(path, lab)
    return lab


def load(gid, refresh=False):
    gid = str(gid)
    path = os.path.join(CACHE, "pff_{}.npz".format(gid))
    if os.path.exists(path) and not refresh:
        return dict(np.load(path))
    meta = _metadata(gid)
    pitch = meta["stadium"]["pitches"][-1]
    L, W = float(pitch["length"]), float(pitch["width"])
    frames = {}
    with bz2.open(os.path.join(DIR, "Tracking Data", gid + ".jsonl.bz2"), "rt") as fh:
        for line in fh:
            f = json.loads(line)
            if f.get("period") in (1, 2) and f.get("videoTimeMs") is not None:
                frames[f["frameNum"]] = f                       # duplicates: keep the last
    order = sorted(frames)
    slots = {}
    for side, team in (("homePlayersSmoothed", 0), ("awayPlayersSmoothed", 1)):
        for n in order:
            for p in frames[n].get(side) or []:
                slots.setdefault((team, p["jerseyNum"]), len(slots))
    t_raw = np.array([frames[n]["videoTimeMs"] / 1000.0 for n in order])
    per_raw = np.array([frames[n]["period"] for n in order], np.int8)
    # 25 Hz grid per period, nearest PFF frame (29.97 Hz)
    grid, per, src = [], [], []
    for h in (1, 2):
        idx = np.flatnonzero(per_raw == h)
        if not len(idx):
            continue
        g = np.arange(t_raw[idx[0]], t_raw[idx[-1]], 1.0 / FPS)
        j = np.clip(np.searchsorted(t_raw[idx], g), 1, len(idx) - 1)
        j = np.where(np.abs(t_raw[idx][j - 1] - g) < np.abs(t_raw[idx][j] - g), j - 1, j)
        grid.append(g)
        per.append(np.full(len(g), h, np.int8))
        src.append(idx[j])
    grid, per, src = np.concatenate(grid), np.concatenate(per), np.concatenate(src)
    P = np.full((len(grid), len(slots), 2), np.nan, np.float32)
    B = np.full((len(grid), 3), np.nan, np.float32)
    for k, s in enumerate(src):
        f = frames[order[s]]
        for side, team in (("homePlayersSmoothed", 0), ("awayPlayersSmoothed", 1)):
            for p in f.get(side) or []:
                if p.get("x") is not None:
                    P[k, slots[(team, p["jerseyNum"])]] = (p["x"], p["y"])
        b = (f.get("balls") or [None])[0]
        if b and b.get("x") is not None:
            B[k] = (b["x"], b["y"], b.get("z") or 0.0)
    # PFF: origin at the centre, metres -> ours: origin at a corner
    P[..., 0] = P[..., 0] * PITCH_L / L + PITCH_L / 2
    P[..., 1] = P[..., 1] * PITCH_W / W + PITCH_W / 2
    B[:, 0] = B[:, 0] * PITCH_L / L + PITCH_L / 2
    B[:, 1] = B[:, 1] * PITCH_W / W + PITCH_W / 2
    team = np.zeros(len(slots), np.int8)
    for (tm, _), i in slots.items():
        team[i] = tm
    out = _layout(P, B[:, :2], per, _labels(gid, grid), team)
    z = np.zeros(len(out["ball"]), np.float32)
    for h, start in enumerate(out["kickoff"], start=1):
        n = int(np.sum(per == h))
        z[start:start + n] = B[per == h, 2]
    out["ball_z"] = np.nan_to_num(z)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **out)
    return out
