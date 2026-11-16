"""Load the provided assets into numpy arrays, cached under cache/ (gitignored).

Everything here is indexed by video frame, i.e. by submission row: index i is ATD frame_id i+1.
Positions are converted to metres; missing observations stay NaN.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "repo")
for path in (REPO, os.path.join(REPO, "starter"), os.path.join(REPO, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402

CACHE = os.path.join(ROOT, "cache")
N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
HOME, AWAY = 0, 1
LABELS = np.array(["H", "A", "D"])
GT_PUBLIC = os.path.join(REPO, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
PUBLIC_WINDOW = (11185, 18684)  # inclusive, from the public GT meta file
KICKOFF_2ND = 113742  # published in FAQ.md: frames 110808..113741 are graded and all D


def halves():
    """[(half, first_frame, last_frame)] graded blocks."""
    return half_video_blocks(GAME)


def half_index():
    """int8 array over frames: 1 or 2 inside a graded half, 0 elsewhere."""
    out = np.zeros(N_FRAMES, np.int8)
    for half, first, last in halves():
        out[first:last + 1] = half
    return out


def load_tracking(refresh=False, source="tactical"):
    """dict with ball (N,2), ball_speed (N,), players (N,S,2), player_speed (N,S), team (S,).

    `source` is "tactical" (the submission timeline) or "broadcast" (its own timeline).
    A few seconds to parse the raw file the first time, then read from cache/.
    """
    name = "tracking.npz" if source == "tactical" else "tracking_{}.npz".format(source)
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and not refresh:
        return dict(np.load(path))

    from baseline import team_of_each_track  # starter/baseline.py, reads the ATD metadata
    from games import source_frames

    paths = asset_paths(source=source)
    n_frames = source_frames(source)
    team = team_of_each_track(paths["atd_meta"])
    n_slots = len(team)
    players = np.full((n_frames, n_slots, 3), np.nan, np.float32)
    ball = np.full((n_frames, 3), np.nan, np.float32)
    with open(paths["atd_raw"]) as handle:
        for line in handle:
            frame_id, groups, ball_group = line.rstrip("\n").split(":")
            i = int(frame_id) - 1
            players[i] = np.array(groups.replace(";", ",").split(","), np.float32).reshape(
                n_slots, 3)
            ball[i] = np.array(ball_group.split(","), np.float32)

    scale = np.array([PITCH_L, PITCH_W], np.float32)
    out = {
        "ball": ball[:, :2] * scale,
        "ball_speed": ball[:, 2],
        "players": players[:, :, :2] * scale,
        "player_speed": players[:, :, 2],
        "team": team.astype(np.int8),
    }
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **out)
    return out


def load_tags():
    """Smart Tagging as a DataFrame: code, team (H/A/None), type, start, end (frames, end
    exclusive)."""
    from load_smart_tagging import load

    rows = []
    for event in load():
        team = (event["tags"].get("Team") or "").strip().lower()
        rows.append({
            "code": (event["code"] or "").strip().upper(),
            "team": "H" if team == "barcelona" else "A" if team == "atletic club" else None,
            "type": (event["tags"].get("Type") or "").strip().lower(),
            "start": event["start_frame"],
            "end": event["end_frame"],
        })
    return pd.DataFrame(rows)


def load_public_gt():
    """Label array over all frames, 'X' outside the public window."""
    return pd.read_csv(GT_PUBLIC)["label"].astype(str).str.strip().str.upper().to_numpy()
