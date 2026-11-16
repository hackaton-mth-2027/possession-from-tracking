"""External training data: the DFL / Sportec open matches (Bassek et al. 2025, CC-BY 4.0).

Seven Bundesliga matches (tools/download_dfl.sh -> external/dfl/), 25 Hz positions of every
player and of the ball, with two flags on every ball frame that are exactly our target:
BallStatus (1 alive, 0 dead) and BallPossession (1 home, 2 away). DFL's own definition of
possession, not Metrica's: used to learn, never to score.

`load(match_id)` returns the match in the format of data.load_tracking (metres, origin at a corner
of the pitch, speeds in m/s), plus labels (0 H, 1 A, 2 D, 3 X), the half of every frame and the
kickoff frame of each half. The two halves are laid end to end with GAP unlabelled frames between
them. Cached as cache/dfl_<match>.npz.

`add_noise` makes a clean DFL match look like our automatic tracking: the ball seen from a camera
(airborne balls land where the line of sight meets the ground), jittered, and missing exactly
where our ball is missing in a randomly chosen stretch of our match; players jittered and
dropped in short fragments.
"""

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

from .data import CACHE, FPS, PITCH_L, PITCH_W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "external", "dfl")
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
GAP = 250
DFL_L, DFL_W = 105.0, 68.0
CAMERA = np.array([53.6, 114.2, 30.1])     # our tactical camera centre (possession/ball3d.py)


def _path(kind, match_id):
    hits = glob.glob(os.path.join(DIR, "DFL_{}_*_DFL-MAT-{}.xml".format(kind, match_id)))
    if not hits:
        raise FileNotFoundError("{} for {}: run tools/download_dfl.sh".format(kind, match_id))
    return hits[0]


def _teams(match_id):
    root = ET.parse(_path("02_01_matchinformation", match_id)).getroot()
    home = guest = None
    for t in root.iter("Team"):
        if t.get("Role") == "home":
            home = t.get("TeamId")
        elif t.get("Role") == "guest":
            guest = t.get("TeamId")
    return home, guest


def _parse(match_id):
    home, guest = _teams(match_id)
    sets = {}      # (section, team, person) -> (frame numbers, x, y, z, speed[, poss, status])
    for _, el in ET.iterparse(_path("04_03_positions_raw_observed", match_id), events=("end",)):
        if el.tag != "FrameSet":
            continue
        team = el.get("TeamId")
        if team in (home, guest, "BALL"):
            fr = el.findall("Frame")
            n = np.array([int(f.get("N")) for f in fr])
            xy = np.array([(float(f.get("X")), float(f.get("Y")), float(f.get("Z", 0)),
                            float(f.get("S"))) for f in fr], np.float32)
            extra = None
            if team == "BALL":
                extra = np.array([(int(f.get("BallPossession", 0)), int(f.get("BallStatus", 0)))
                                  for f in fr], np.int8)
            sets[(el.get("GameSection"), team, el.get("PersonId"))] = (n, xy, extra)
        el.clear()
    return home, guest, sets


def load(match_id, refresh=False):
    path = os.path.join(CACHE, "dfl_{}.npz".format(match_id))
    if os.path.exists(path) and not refresh:
        return dict(np.load(path))
    home, guest, sets = _parse(match_id)
    sections = [s for s in ("firstHalf", "secondHalf") if (s, "BALL") in
                {(k[0], k[1]) for k in sets}]
    persons = sorted({k[2] for k in sets if k[1] != "BALL"})
    team_of = {k[2]: (0 if k[1] == home else 1) for k in sets if k[1] != "BALL"}
    # frame range of each half from the ball
    spans, offset = [], 0
    for s in sections:
        n, _, _ = next(v for k, v in sets.items() if k[0] == s and k[1] == "BALL")
        spans.append((s, int(n.min()), int(n.max()), offset))
        offset += int(n.max() - n.min() + 1) + GAP
    total = offset - GAP
    players = np.full((total, len(persons), 2), np.nan, np.float32)
    speed = np.full((total, len(persons)), np.nan, np.float32)
    ball = np.full((total, 3), np.nan, np.float32)
    ball_speed = np.full(total, np.nan, np.float32)
    labels = np.full(total, 3, np.int8)
    half = np.zeros(total, np.int8)
    kickoff = []
    slot = {p: i for i, p in enumerate(persons)}
    for h, (s, lo, hi, off) in enumerate(spans, start=1):
        half[off:off + hi - lo + 1] = h
        kickoff.append(off)
        for (sec, team, person), (n, v, extra) in sets.items():
            if sec != s:
                continue
            i = n - lo + off
            # DFL: origin at the centre, metres -> ours: origin at a corner
            x = v[:, 0] * PITCH_L / DFL_L + PITCH_L / 2
            y = v[:, 1] * PITCH_W / DFL_W + PITCH_W / 2
            if team == "BALL":
                ball[i] = np.column_stack([x, y, v[:, 2]])
                ball_speed[i] = v[:, 3] / 3.6
                poss, status = extra[:, 0], extra[:, 1]
                labels[i] = np.where(status == 0, 2, np.where(poss == 1, 0, np.where(
                    poss == 2, 1, 3)))
            else:
                players[i, slot[person]] = np.column_stack([x, y])
                speed[i, slot[person]] = v[:, 3] / 3.6          # km/h -> m/s
    out = {"players": players, "player_speed": speed, "ball": ball[:, :2], "ball_z": ball[:, 2],
           "ball_speed": ball_speed, "team": np.array([team_of[p] for p in persons], np.int8),
           "labels": labels, "half": half, "kickoff": np.array(kickoff)}
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **out)
    return out


def _fragments(rng, n, rate, mean_len):
    """Boolean (n,) mask of dropped frames: about `rate` of them, in runs of ~mean_len frames."""
    keep = np.ones(n, bool)
    n_runs = int(rate * n / mean_len)
    for s in rng.integers(0, n, n_runs):
        keep[s:s + int(rng.exponential(mean_len)) + 1] = False
    return ~keep


def _state_mask(rng, labels, pools):
    """Missing-ball mask following the game state: each run of live (or dead) frames takes a
    random stretch of the same length from our live (or dead) frames' missing mask."""
    n = len(labels)
    dead = ~np.isin(labels, (0, 1))
    out = np.zeros(n, bool)
    b = np.flatnonzero(np.diff(dead.astype(np.int8)) != 0) + 1
    for s, e in zip(np.r_[0, b], np.r_[b, n]):
        pool = pools[1] if dead[s] else pools[0]
        L = e - s
        if len(pool) > L:
            k = rng.integers(0, len(pool) - L)
            out[s:e] = pool[k:k + L]
        else:
            out[s:e] = np.resize(pool, L)
    return out


def add_noise(m, our_ball_missing, seed=0, ball_sigma=0.5, player_sigma=0.3,
              player_drop=0.07, state_pools=None, ball_white=None, player_smooth=False,
              air_miss=0.0, air_z=1.5):
    """A copy of a DFL match (dict from load) that looks like our automatic tracking.
    `state_pools` = (missing mask over our live frames, over our dead frames): the ball then goes
    missing as it does in our match in the same game state (72% of dead-ball frames, 11% of live
    ones, by a first model's decoding) instead of at random places.
    Calibration knobs (defaults = the original model, same random draws): `ball_white` per-frame
    ball jitter (default 0.5 x ball_sigma; our real ball is much smoother), `player_smooth` slow
    drifting player errors instead of per-frame jitter, `air_miss` extra probability that the ball
    is lost while higher than `air_z` m (our camera loses airborne balls)."""
    rng = np.random.default_rng(seed)
    n = len(m["ball"])
    out = {k: v.copy() for k, v in m.items()}
    # ball seen from the camera: airborne balls are drawn where the line of sight hits the ground
    z = np.nan_to_num(m["ball_z"])
    k = CAMERA[2] / np.maximum(CAMERA[2] - z, 1e-3)
    ball = CAMERA[:2] + k[:, None] * (m["ball"] - CAMERA[:2])
    # error: a slow wander (random walk with its 5 s trend removed) plus per-frame jitter
    walk = np.cumsum(rng.normal(0, 1, (n, 2)), axis=0)
    walk -= np.column_stack([np.convolve(walk[:, j], np.ones(125) / 125, mode="same")
                             for j in range(2)])
    walk /= max(1e-6, float(walk.std()))
    white = 0.5 * ball_sigma if ball_white is None else ball_white
    ball = ball + 0.7 * ball_sigma * walk + rng.normal(0, 1.0, (n, 2)) * white
    if state_pools is not None:
        miss = _state_mask(rng, m["labels"], state_pools)
    else:
        # missing exactly where ours is missing, in a random stretch of our match
        start = rng.integers(0, len(our_ball_missing) - n) if len(our_ball_missing) > n else 0
        miss = our_ball_missing[start:start + n]
        if len(miss) < n:
            miss = np.resize(miss, n)
    ball[miss] = np.nan
    out["ball"] = ball.astype(np.float32)
    out["ball_speed"] = np.where(miss, np.nan, m["ball_speed"]).astype(np.float32)
    e = rng.normal(0, 1.0, m["players"].shape)
    if player_smooth:
        e = np.cumsum(e, axis=0)
        e -= np.stack([np.stack([np.convolve(e[:, i, j], np.ones(125) / 125, mode="same")
                                 for j in range(2)], axis=1) for i in range(e.shape[1])], axis=1)
        e /= max(1e-6, float(np.nanstd(e)))
    P = m["players"] + (player_sigma * e).astype(np.float32)
    for j in range(P.shape[1]):
        drop = _fragments(rng, n, player_drop, 50)
        P[drop, j] = np.nan
    out["players"] = P
    out["player_speed"] = np.where(np.isnan(P[:, :, 0]), np.nan, m["player_speed"])
    if air_miss > 0:
        lost = (z > air_z) & (rng.random(n) < air_miss)
        out["ball"][lost] = np.nan
        out["ball_speed"][lost] = np.nan
    return out


# -- label variants: DFL's per-touch possession is not Metrica's controlled possession ---------

def _half_starts(match_id):
    """UTC timestamp of the first ball frame of each half, from the positions file."""
    import subprocess
    from datetime import datetime
    out = subprocess.run(["grep", "-A1", 'TeamId="BALL"', _path("04_03_positions_raw_observed",
                                                              match_id)],
                         capture_output=True, text=True).stdout.splitlines()
    starts = []
    for line in out:
        if line.startswith("<Frame "):
            t = line.split(' T="')[1].split('"')[0]
            starts.append(datetime.fromisoformat(t).timestamp())
    return starts


def controlled_events(match_id, m):
    """[(frame, team 0/1)] of controlled ball actions (passes, receptions / ball claims, shots,
    other actions), on the timeline of load()."""
    from datetime import datetime
    home, _ = _teams(match_id)
    starts = _half_starts(match_id)
    root = ET.parse(_path("03_02_events_raw", match_id)).getroot()
    out = []
    for e in root.iter("Event"):
        t = datetime.fromisoformat(e.get("EventTime")).timestamp()
        for c in e.iter():
            if c.get("Team") and c.tag in ("Play", "BallClaiming", "ShotAtGoal",
                                           "OtherBallAction"):
                h = 2 if len(starts) > 1 and t >= starts[1] - 1 else 1
                f = int(m["kickoff"][h - 1] + round((t - starts[h - 1]) * FPS))
                if 0 <= f < len(m["labels"]) and m["half"][f] == h:
                    out.append((f, 0 if c.get("Team") == home else 1))
                break
    return sorted(out)


def labels_events(match_id, m):
    """Controlled possession: the team changes only between two controlled actions of different
    teams, at the start of the new team's last run of tracking ownership before its action (the
    tracking flag gives the timing, the events say which changes are real). Dead ball as DFL."""
    lab = m["labels"].copy()
    owner = np.where(np.isin(lab, (0, 1)), lab, -1)
    ev = controlled_events(match_id, m)
    team = np.full(len(lab), -1, np.int64)
    for (f1, t1), (f2, t2) in zip(ev, ev[1:]):
        change = f2
        if t1 != t2:
            k = f2
            while k > f1 and owner[k] == t2:
                k -= 1
            change = k + 1 if k < f2 else f2
        team[f1:change] = t1
        team[change:f2] = t2
    if ev:
        team[ev[-1][0]:] = ev[-1][1]
    alive = np.isin(lab, (0, 1))
    use = alive & (team >= 0)
    lab[use] = team[use]
    return lab


def team_actions(match_id, m):
    """[(frame, team 0/1, kind)] of the teams' ball actions on the timeline of load(); kind is the
    event's tag: Play (a pass), ShotAtGoal, BallClaiming, OtherBallAction."""
    from datetime import datetime
    home, _ = _teams(match_id)
    starts = _half_starts(match_id)
    root = ET.parse(_path("03_02_events_raw", match_id)).getroot()
    out = []
    for e in root.iter("Event"):
        t = datetime.fromisoformat(e.get("EventTime")).timestamp()
        for c in e.iter():
            if c.get("Team") and c.tag in ("Play", "BallClaiming", "ShotAtGoal",
                                           "OtherBallAction"):
                h = 2 if len(starts) > 1 and t >= starts[1] - 1 else 1
                f = int(m["kickoff"][h - 1] + round((t - starts[h - 1]) * FPS))
                if 0 <= f < len(m["labels"]) and m["half"][f] == h:
                    out.append((f, 0 if c.get("Team") == home else 1, c.tag))
                break
    return sorted(out)


def labels_control(match_id, m, rule):
    """DFL's possession under the challenge's controlled-possession rule
    (possession/metrica_open.CONTROL_RULES): a spell won in open play counts only if its team
    plays a pass or a shot, or keeps the ball long enough; otherwise it goes back to the team that
    had the ball before. Spells starting at a restart always count."""
    from .metrica_open import CONTROL_RULES
    seconds, passes = CONTROL_RULES[rule]["seconds"], CONTROL_RULES[rule]["passes"]
    acts = team_actions(match_id, m)
    f_ = np.array([a[0] for a in acts])
    t_ = np.array([a[1] for a in acts])
    k_ = np.array([a[2] in ("Play", "ShotAtGoal") for a in acts])
    lab = m["labels"].copy()
    live = np.isin(lab, (0, 1))
    edges = np.flatnonzero(np.diff(np.r_[-1, lab, -1]) != 0)
    for s, e in zip(edges[:-1], edges[1:]):
        k = lab[s]
        if k not in (0, 1) or s == 0 or not live[s - 1] or lab[s - 1] == k:
            continue
        mine = (f_ >= s) & (f_ < e) & (t_ == k)
        if (e - s) >= seconds * FPS or np.sum(mine & k_) >= passes:
            continue
        lab[s:e] = lab[s - 1]
    return lab


def labels_sandwich(m, max_s):
    """DFL's possession with opponent spells shorter than max_s seconds removed when the same
    team has the ball on both sides (live play only): deflections and lost-and-won-back balls."""
    lab = m["labels"].copy()
    n = len(lab)
    b = np.flatnonzero(np.diff(lab) != 0) + 1
    starts = np.r_[0, b]
    ends = np.r_[b, n]
    vals = lab[starts]
    for i in range(1, len(vals) - 1):
        if vals[i] in (0, 1) and vals[i - 1] == vals[i + 1] and vals[i - 1] in (0, 1) \
                and vals[i] != vals[i - 1] and ends[i] - starts[i] < max_s * FPS:
            lab[starts[i]:ends[i]] = vals[i - 1]
    return lab


def labels_variant(match_id, m, target):
    if target == "touch":
        return m["labels"]
    if target in ("lite", "strict"):
        return labels_control(match_id, m, target)
    if target == "events":
        return labels_events(match_id, m)
    if target.startswith("sandwich"):
        return labels_sandwich(m, float(target[len("sandwich"):]))
    raise ValueError(target)
