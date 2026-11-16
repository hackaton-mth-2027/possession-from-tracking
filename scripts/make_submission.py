"""The submission: frame models + change and boundary detectors + Viterbi, over the whole match.

    .venv/bin/python scripts/make_submission.py [--name dfl_bag_st_dflcd]

1. Frame probabilities P(H), P(A), P(D), each model on its own random noise draws:
   - team: the home share averaged within two families, weighted NET_WEIGHT: 9 gradient-boosting
     models learned on the 7 DFL open matches with the extra features of
     possession/features_extra.py (scripts/team_features.py; 6 with the ball missing at random
     places, 3 with it missing as ours does in each game state), and 16 temporal networks over
     +-5 s (scripts/dfl_tcn.py): NETS (seeds and variants) on DFL, and the same 8 on DFL + the 3
     Metrica open matches labelled by the challenge's controlled-possession rule (MET_NETS)
   - dead ball: average of the PFF_SEEDS models learned on DFL + the 64 PFF World Cup matches,
     whose dead ball follows the events (ball out to restart) like the provider's
     (scripts/pff_train.py, needs tools/download_pff.sh)
   Each is trained first if missing. One model's score moves by ~0.02-0.03 with its noise draw;
   the averages do not.
2. Change probabilities from the change detector learned on the DFL matches (scripts/dfl_change.py),
   dead-ball boundary probabilities from the detector learned on the PFF matches
   (scripts/dead_change.py), each trained first if missing.
3. Viterbi per half, team-switch cost max(4, 28 - 23 * p_change), cost of entering or leaving dead
   ball max(4, 24 - 12 * p_boundary).
Model and decoder choices were made on the Metrica validation set (scripts/validate_clean.py and
the experiment scripts named below; TECHNICAL.md). The dead-ball training data follows the
public ground truth's convention (TECHNICAL.md, section 1); otherwise the official window is a
check. Nothing is fitted on this match's labels: the repository regenerates the file from public
data plus the match data.
Writes outputs/submissions/<name>.csv (X outside the graded halves) and validates it.
"""

import argparse
import os
import subprocess
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from possession import data, decode  # noqa: E402
import change_detector as cd  # noqa: E402


def write(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


# decoder (scripts/validate_clean.py, scripts/dead_change.py): on a plateau of team costs 20-32
# with a bonus of 0.8-1x the cost; against our first costs (48 / 24 / 128 in 5-frame steps) +0.044
# [+0.035, +0.055] on the Metrica set. Dead cost 24 with the boundary detector: +0.010 against 32
# alone, official macroF1 0.917 -> 0.930.
TEAM_COST = 28.0
BONUS = 23.0
DEAD_COST = 24.0
DEAD_BONUS = 12.0
# decoding in steps of STEP frames; 3, 5 or 10 do not help at these costs, so frame by frame
STEP = 1
# team (scripts/team_features.py bag): the two families half each against the 9 DFL boosting
# models alone: Metrica +0.023 [+0.014, +0.032], official window 0.653 -> 0.690
BOOST = [(s, False) for s in (0, 2, 4, 6, 8, 10)] + [(s, True) for s in (0, 2, 4)]
NETS = {"dfl_tcn_all": [], "dfl_tcn_all_s1": ["--seed", "1"], "dfl_tcn_all_s2": ["--seed", "2"],
        "dfl_tcn_v_dil128": ["--max-dil", "128"], "dfl_tcn_v_ep12": ["--epochs", "12"],
        "dfl_tcn_v_st": ["--noise", "st"], "dfl_tcn_v_ch128": ["--channels", "128"],
        "dfl_tcn_v_extra": ["--extra"]}
NET_WEIGHT = 0.5
# the same networks also trained on Metrica's 3 open matches, labelled by the challenge's
# "controlled possession" rule (possession/metrica_open.CONTROL_RULES "lite"); leave one match out,
# added to the 8: Metrica (lite labels) +0.010 [+0.004, +0.016], official window equal
MET_NETS = {stem + "_met3": args + ["--metrica", "M1,M2,M3", "--met-weight", "3"]
            for stem, args in NETS.items()}
# dead ball from DFL + PFF (scripts/pff_train.py dead, dead_change.py choice)
PFF_SEEDS = (0, 2, 4)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", default="dfl_bag_st_dflcd")
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    boost = [boost_member(s, st) for s, st in BOOST]
    nets = [net_member(stem, args) for stem, args in NETS.items()] + \
        [net_member(stem, args, {"METRICA_LABELS": "lite"}) for stem, args in MET_NETS.items()]
    dead = np.mean([load(pff_member(s))[:, 2] for s in PFF_SEEDS], axis=0)
    home = (1 - NET_WEIGHT) * home_share(boost) + NET_WEIGHT * home_share(nets)
    proba = np.column_stack([(1 - dead) * home, (1 - dead) * (1 - home), dead])
    np.save(os.path.join(data.CACHE, "bag_submission.npy"), proba.astype(np.float32))
    p_change = dfl_change_probabilities()
    p_dead = dead_change_probabilities()
    out = np.full(data.N_FRAMES, "X", dtype="<U1")
    for half, first, last in data.halves():
        if half == 2:
            # graded from 110808 but kicked off at 113742 (DATA.md, FAQ): all dead ball before
            out[first:data.KICKOFF_2ND] = "D"
            first = data.KICKOFF_2ND
        out[first:last + 1] = data.LABELS[decode_steps(proba[first:last + 1],
                                                       p_change[first:last + 1],
                                                       p_dead[first:last + 1])]
    path = os.path.join(ROOT, "outputs", "submissions", args.name + ".csv")
    write(path, out)
    print("wrote", os.path.relpath(path, ROOT))
    subprocess.run([sys.executable, os.path.join(ROOT, "repo", "evaluation",
                                                 "validate_submission.py"), path], check=False)


def load(name):
    return np.load(os.path.join(data.CACHE, name)).astype(np.float64)


def home_share(names):
    return np.mean([p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-9) for p in map(load, names)],
                   axis=0)


def run(script, *args, env=None):
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script)] + list(args),
                   check=True, env=dict(os.environ, **(env or {})))


def boost_member(seed, state_noise):
    name = "team_extra{}_s{}_ours.npy".format("_st" if state_noise else "", seed)
    if not os.path.exists(os.path.join(data.CACHE, name)):
        run("team_features.py", "proba")
    return name


def net_member(stem, args, env=None):
    name = stem + ".npy"
    if not os.path.exists(os.path.join(data.CACHE, name)):
        run("dfl_tcn.py", "--all", "--out", name, *args, env=env)
    return name


def pff_member(seed):
    name = "pff_proba_st_s{}.npy".format(seed)
    if not os.path.exists(os.path.join(data.CACHE, name)):
        run("pff_train.py", "proba", str(seed))
    return name


def decode_steps(proba, p_change, p_dead):
    """Viterbi over steps of STEP frames (the remainder at the end takes the last step's label)."""
    n = len(proba) // STEP * STEP
    lp = np.log(np.clip(proba[:n], 1e-6, 1.0)).reshape(-1, STEP, 3).sum(1)
    pcs = p_change[:n].reshape(-1, STEP).max(1)
    tc = np.maximum(cd.MIN_COST, TEAM_COST - BONUS * pcs)
    dd = np.maximum(cd.MIN_COST, DEAD_COST - DEAD_BONUS * p_dead[:n].reshape(-1, STEP).max(1))
    steps = decode.viterbi_tv(np.exp(lp - np.logaddexp.reduce(lp, 1, keepdims=True)), tc, dd, dd,
                              eps=1e-300)
    out = np.empty(len(proba), np.int8)
    out[:n] = np.repeat(steps, STEP)
    out[n:] = steps[-1]
    return out


def dead_change_probabilities():
    path = os.path.join(data.CACHE, "pdead_change_pff.npy")
    if not os.path.exists(path):
        run("dead_change.py", "proba")
    return load("pdead_change_pff.npy")


def dfl_change_probabilities():
    path = os.path.join(data.CACHE, "dfl_pchange_touch.npy")
    if not os.path.exists(path):
        run("dfl_change.py", "--target", "touch")
    return load("dfl_pchange_touch.npy")


if __name__ == "__main__":
    main()
