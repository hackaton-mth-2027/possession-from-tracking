"""Score changes to the submission's team estimate from cached probabilities.

    .venv/bin/python scripts/team_mix.py STEM [STEM ...]

The submission's team: the home share of the 9 boosting models (make_submission.BOOST) and of
the 16 networks (make_submission.NETS and MET_NETS, the latter scored on Metrica by their
leave-one-match-out versions <stem>_lomo), half each. Each STEM is a cached set of probabilities
(cache/STEM.npy for our match, STEM_M1.npy ... for the Metrica games, as scripts/dfl_tcn.py
writes them). Reported, paired on the Metrica validation set against the submission, plus the
official window:
  - each STEM alone, against the first network alone (same architecture, seed 0)
  - the network family with the STEMs added, and with the STEMs instead
Dead ball, boundary detector and decoder as the submission.
"""

import os
import re
import sys
import warnings

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
from possession import data, features  # noqa: E402
from dfl_pretrain import DROP, our_missing_mask  # noqa: E402
import validate_clean as vc  # noqa: E402
from dead_change import detector, set_pdc  # noqa: E402
from pff_train import official_paired  # noqa: E402
from team_features import with_sources  # noqa: E402
from team_retest import FULL  # noqa: E402
import make_submission as ms  # noqa: E402

GAMES = ("M1", "M2", "M3")


def source(stem, ours=""):
    """Our match: cache/<stem><ours>.npy (boosting members: "_ours"); Metrica: <stem>_M1.npy..."""
    d = {g: np.load(os.path.join(data.CACHE, "{}_{}.npy".format(stem, g))).astype(np.float64)
         for g in GAMES}
    d["ours"] = np.load(os.path.join(data.CACHE, stem + ours + ".npy")).astype(np.float64)
    return d


def home(srcs, g):
    return np.mean([s[g][:, 0] / np.maximum(s[g][:, 0] + s[g][:, 1], 1e-9) for s in srcs], 0)


def mixed(boost, nets, w=ms.NET_WEIGHT):
    out = {}
    for g in GAMES + ("ours",):
        h = (1 - w) * home(boost, g) + w * home(nets, g)
        out[g] = np.column_stack([h, 1 - h, np.zeros_like(h)])
    return out


def samples():
    miss = our_missing_mask()
    ours, _ = features.build()
    cols = [c for c in ours.columns if not re.search(DROP, c)]
    dead = [joblib.load(os.path.join(data.CACHE, "pff_dflpff_st_s{}.joblib".format(s)))
            for s in ms.PFF_SEEDS]
    names = ["dfl_model.joblib"]
    S = vc.Samples(plain=vc.load_models(names), state=dead)
    set_pdc(S, detector("pff", miss, cols), miss, cols, ours)
    return S, names, miss


def main():
    warnings.filterwarnings("ignore")
    stems = sys.argv[1:]
    S, names, miss = samples()
    boost = [source("team_extra{}_s{}".format("_st" if st else "", s), "_ours")
             for s, st in ms.BOOST]
    nets = [source(stem) for stem in ms.NETS] + [source(stem[:-len("_met3")] + "_lomo")
                                                 for stem in ms.MET_NETS]
    new = [source(stem) for stem in stems]

    def score(srcs, cfg=FULL):
        return vc.evaluate(with_sources(S, names, srcs, miss), cfg)

    ref = score([mixed(boost, nets)])
    vc.report("submission (9 boosting + {} networks)".format(len(nets)), ref, ref)
    one = score([nets[0]])
    vc.report("  network dfl_tcn_all alone", one, one)
    for stem, src in zip(stems, new):
        res = score([src])
        vc.report("  {} alone (vs dfl_tcn_all alone)".format(stem), one, res)
        print("      official: " + official_paired(one, res), flush=True)
    for name, family in (("networks + " + " ".join(stems), nets + new),
                         ("networks = " + " ".join(stems), new)):
        res = score([mixed(boost, family)])
        vc.report(name, ref, res)
        print("    official: " + official_paired(ref, res), flush=True)


if __name__ == "__main__":
    main()
