"""External validation data with REAL automatic tracking: SkillCorner open data, 20 A-League
2024/25 matches (MIT, github.com/SkillCorner/opendata, tools/download_skillcorner.sh ->
external/skillcorner/).

Broadcast tracking at 10 fps: the players the camera sees are detected, the others extrapolated
by SkillCorner (`is_detected`); the ball with a detected flag. Unlike the DFL, Metrica and PFF
copies used elsewhere, none of its noise comes from possession/dfl.add_noise: it is what an
automatic tracker produced. Never used for training.

Labels per frame (0 H, 1 A, 2 D, 3 X), from SkillCorner's own files:
  dead ball   frames outside every phase of play (`<id>_phases_of_play.csv`: phases exist only
              while the ball is in play)
  possession  the team of the last player possession (`<id>_dynamic_events.csv`, event_type
              player_possession: a player controlling the ball, from frame_start); between two
              player possessions the ball stays with the last team, as in CHALLENGE.md
`labels(gid, rule)` applies possession/metrica_open.CONTROL_RULES: a spell won in open play counts
only if its team passes or shoots, or keeps the ball long enough.
`load(gid)` returns the same dict as possession/dfl.load, resampled to our 25 Hz.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from .data import CACHE, FPS, PITCH_L, PITCH_W
from .metrica_open import CONTROL_RULES, _layout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "external", "skillcorner")
SRC_FPS = 10


def games():
    """Ids of the matches whose four files are present."""
    ids = []
    for path in sorted(glob.glob(os.path.join(DIR, "*_tracking_extrapolated.jsonl"))):
        gid = os.path.basename(path).split("_")[0]
        if all(os.path.exists(os.path.join(DIR, gid + s)) for s in
               ("_match.json", "_dynamic_events.csv", "_phases_of_play.csv")):
            ids.append(gid)
    return ids


def metadata(gid):
    with open(os.path.join(DIR, "{}_match.json".format(gid))) as fh:
        return json.load(fh)


def events(gid):
    """The match's player possessions, in time order."""
    ev = pd.read_csv(os.path.join(DIR, "{}_dynamic_events.csv".format(gid)), low_memory=False)
    ev = ev[ev.event_type == "player_possession"]
    return ev.sort_values(["frame_start", "frame_end"], kind="stable").reset_index(drop=True)


def phases(gid):
    return pd.read_csv(os.path.join(DIR, "{}_phases_of_play.csv".format(gid)))


def raw(gid, refresh=False):
    """The tracking as provided, one row per 10 fps frame of periods 1 and 2: players (n, slots,
    2) and ball (n, 3) in SkillCorner's metres from the centre, `detected` (n, slots), `ball_det`,
    `frame`, `period`, `group` (the tracking's own team in possession: 0 home, 1 away, -1 none),
    `team` per slot. Cached as cache/sc_raw_<gid>.npz."""
    gid = str(gid)
    path = os.path.join(CACHE, "sc_raw_{}.npz".format(gid))
    if os.path.exists(path) and not refresh:
        return dict(np.load(path))
    meta = metadata(gid)
    home = meta["home_team"]["id"]
    slot, team = {}, []
    for p in meta["players"]:
        slot[p["id"]] = len(slot)
        team.append(0 if p["team_id"] == home else 1)
    rows = []
    with open(os.path.join(DIR, "{}_tracking_extrapolated.jsonl".format(gid))) as fh:
        for line in fh:
            f = json.loads(line)
            if f.get("period") in (1, 2):
                rows.append(f)
    n = len(rows)
    P = np.full((n, len(slot), 2), np.nan, np.float32)
    det = np.zeros((n, len(slot)), bool)
    B = np.full((n, 3), np.nan, np.float32)
    bdet = np.zeros(n, bool)
    frame = np.zeros(n, np.int64)
    period = np.zeros(n, np.int8)
    group = np.full(n, -1, np.int8)
    for k, f in enumerate(rows):
        frame[k], period[k] = f["frame"], f["period"]
        for p in f.get("player_data") or []:
            j = slot.get(p["player_id"])
            if j is not None and p.get("x") is not None:
                P[k, j] = (p["x"], p["y"])
                det[k, j] = bool(p.get("is_detected"))
        b = f.get("ball_data") or {}
        if b.get("x") is not None:
            B[k] = (b["x"], b["y"], b.get("z") or 0.0)
            bdet[k] = bool(b.get("is_detected"))
        g = (f.get("possession") or {}).get("group")
        group[k] = {"home team": 0, "away team": 1}.get(g, -1)
    out = {"players": P, "detected": det, "ball": B, "ball_det": bdet, "frame": frame,
           "period": period, "group": group, "team": np.array(team, np.int8)}
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **out)
    return out


def _index(r, frames):
    return np.searchsorted(r["frame"], np.asarray(frames))


def _labels10(gid, r, rule=None):
    """Label of each 10 fps row of raw(gid)."""
    meta = metadata(gid)
    home = meta["home_team"]["id"]
    n = len(r["frame"])
    ev, ph = events(gid), phases(gid)
    live = np.zeros(n, bool)
    for a, b in zip(_index(r, ph.frame_start), _index(r, ph.frame_end)):
        live[a:b] = True
    start = _index(r, ev.frame_start)
    team = np.where(ev.team_id.to_numpy() == home, 0, 1)
    owner = np.full(n, -1, np.int64)
    for k in range(len(ev)):
        owner[start[k]:(start[k + 1] if k + 1 < len(ev) else n)] = team[k]
    # a restart: until the first player possession of a stretch of play, the ball belongs to the
    # team that took the restart (the receiver's team, or the other one if he intercepted it)
    taker = np.where(ev.start_type.astype(str).str.contains("interception|recovery").to_numpy(),
                     1 - team, team)
    edges = np.flatnonzero(np.diff(np.r_[False, live, False]))
    for a, b in zip(edges[::2], edges[1::2]):
        k = np.searchsorted(start, a)
        if k < len(ev) and start[k] < b:
            owner[a:start[k]] = taker[k]
    lab = np.where(owner >= 0, owner, 3).astype(np.int8)
    lab[~live] = 2
    if rule is not None:
        lab = _control(lab, ev, start, team, rule)
    return lab


def _control(lab, ev, start, team, rule):
    """possession/metrica_open._control for SkillCorner's events: a spell won in open play is kept
    only if its team completes a pass or shoots, or keeps the ball CONTROL_RULES[rule]["seconds"];
    otherwise it goes back to the team before."""
    seconds, passes = CONTROL_RULES[rule]["seconds"], CONTROL_RULES[rule]["passes"]
    good = ((ev.end_type == "pass") & (ev.pass_outcome == "successful")).to_numpy()
    shot = (ev.end_type == "shot").to_numpy()
    end = start + (ev.frame_end - ev.frame_start).to_numpy()      # the pass leaves at frame_end
    lab = lab.copy()
    live = np.isin(lab, (0, 1))
    edges = np.flatnonzero(np.diff(np.r_[-1, lab, -1]) != 0)
    for s, e in zip(edges[:-1], edges[1:]):
        k = lab[s]
        if k not in (0, 1) or s == 0 or not live[s - 1] or lab[s - 1] == k:
            continue
        mine = (end >= s) & (end < e) & (team == k)
        if (e - s) >= seconds * SRC_FPS or np.sum(mine & good) >= passes or np.any(mine & shot):
            continue
        lab[s:e] = lab[s - 1]
    return lab


def labels(gid, rule):
    """Laid-out 25 Hz labels of match `gid` under CONTROL_RULES[rule] ("events": every player
    possession), cached as cache/sc_<gid>_<rule>.npy."""
    gid = str(gid)
    path = os.path.join(CACHE, "sc_{}_{}.npy".format(gid, rule))
    if os.path.exists(path):
        return np.load(path)
    r = raw(gid)
    lab10 = _labels10(gid, r, None if rule == "events" else rule)
    per, lab = [], []
    for h, idx, j, _ in _grid(r):
        per.append(np.full(len(j), h, np.int8))
        lab.append(lab10[idx[j]])
    per, lab = np.concatenate(per), np.concatenate(lab)
    out = _layout(np.zeros((len(lab), 1, 2), np.float32), np.zeros((len(lab), 2), np.float32),
                  per, lab, np.zeros(1))["labels"]
    np.save(path, out)
    return out


def _grid(r):
    """Per period: (period, its 10 fps rows, for each 25 Hz frame the row in force, the weight of
    the next row)."""
    for h in (1, 2):
        idx = np.flatnonzero(r["period"] == h)
        if len(idx) < 2:
            continue
        t = (r["frame"][idx] - r["frame"][idx[0]]) / SRC_FPS
        grid = np.arange(0, t[-1], 1.0 / FPS)
        j = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(idx) - 2)
        w = np.clip((grid - t[j]) / np.maximum(t[j + 1] - t[j], 1e-9), 0, 1).astype(np.float32)
        yield h, idx, j, w


def _resample(r, lab10):
    """10 fps -> 25 Hz per period: positions interpolated linearly between the two source frames
    (the valid one where only one is), labels from the source frame in force. Returns players,
    ball (x, y, z; NaN where the ball is not detected), period, labels."""
    P, B, per, lab = [], [], [], []
    ball = np.where(r["ball_det"][:, None], r["ball"], np.nan)
    for h, idx, j, w in _grid(r):
        def lerp(a):
            a0, a1 = a[idx[j]], a[idx[j + 1]]
            wv = w.reshape((-1,) + (1,) * (a.ndim - 1))
            return np.where(np.isnan(a0), a1, np.where(np.isnan(a1), a0, (1 - wv) * a0 + wv * a1))
        P.append(lerp(r["players"]))
        B.append(lerp(ball))
        per.append(np.full(len(j), h, np.int8))
        lab.append(lab10[idx[j]])
    return np.concatenate(P), np.concatenate(B), np.concatenate(per), np.concatenate(lab)


def load(gid, refresh=False):
    """Same dict as possession/dfl.load, at 25 Hz in our pitch coordinates; labels: every player
    possession ("events"). The ball only where SkillCorner detected it (elsewhere it is their
    extrapolation); every player, detected or extrapolated. Cached as cache/sc_<gid>.npz."""
    gid = str(gid)
    path = os.path.join(CACHE, "sc_{}.npz".format(gid))
    if os.path.exists(path) and not refresh:
        return dict(np.load(path))
    meta = metadata(gid)
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    r = raw(gid)
    P, B, per, lab = _resample(r, _labels10(gid, r))
    P[..., 0] = P[..., 0] * PITCH_L / L + PITCH_L / 2
    P[..., 1] = P[..., 1] * PITCH_W / W + PITCH_W / 2
    B[:, 0] = B[:, 0] * PITCH_L / L + PITCH_L / 2
    B[:, 1] = B[:, 1] * PITCH_W / W + PITCH_W / 2
    out = _layout(P, B[:, :2], per, lab, r["team"])
    z = np.zeros(len(out["ball"]), np.float32)
    for h, k0 in enumerate(out["kickoff"], start=1):
        z[k0:k0 + int(np.sum(per == h))] = B[per == h, 2]
    out["ball_z"] = np.nan_to_num(z)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **out)
    return out
