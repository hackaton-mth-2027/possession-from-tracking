"""Team-side variants rejected earlier, re-scored with the current pipeline on the corrected Metrica
labels: dead ball from the 3 DFL + PFF models, boundary detector, decoder 28 / 23 / 24 (bonus 12).

    .venv/bin/python scripts/team_retest.py
    .venv/bin/python scripts/team_retest.py tcn [FILE ...]   # temporal network(s) in the team bag

Only the team estimate differs. Single-model variants are compared, paired, with the plain DFL
model of the same noise draw (seed 0) as the only team model; bag variants with the submission's
9-model team bag.
"""

import copy
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
from dead_change import SUBMISSION, detector, set_pdc  # noqa: E402
from pff_train import official_paired  # noqa: E402

FULL = dict(SUBMISSION, dead=24.0, dead_bonus=12.0)
BAG = vc.PLAIN + vc.STATE
EXTRA = ["dfl_model_s{}.joblib".format(s) for s in (12, 14, 16, 18, 20, 22)]
SINGLE = {"target: opponent spells < 2 s removed": "dfl_model_sandwich2.joblib",
          "target: opponent spells < 1 s removed": "dfl_model_sandwich1.joblib",
          "target: controlled actions (events)": "dfl_model_events.joblib",
          "mirror copies": "dfl_model_mir.joblib"}


def subset(S, names, keep):
    """The same samples with only the team models in `keep`."""
    idx = [names.index(k) for k in keep]
    T = copy.copy(S)
    T.units = {"Metrica": [(lab, [pp[i] for i in idx], ps, pc)
                           for lab, pp, ps, pc in S.units["Metrica"]],
               "match": [(a, b, [pp[i] for i in idx], ps, pc)
                         for a, b, pp, ps, pc in S.units["match"]]}
    return T


def with_tcn(S, names, keep=None, stems=("dfl_tcn_all",), copies=1):
    """Team from the temporal networks' cached probabilities (cache/<stem>.npy for our match,
    <stem>_<game>.npy for Metrica), each repeated `copies` times, after the boosting models in
    `keep` (none: the networks alone)."""
    T = subset(S, names, keep or [names[0]])
    nets = {g: [np.load(os.path.join(data.CACHE, "{}_{}.npy".format(st, g))).astype(np.float64)
                for st in stems] for g in ("M1", "M2", "M3")}
    nets["ours"] = [np.load(os.path.join(data.CACHE, st + ".npy")).astype(np.float64)
                    for st in stems]
    i, units = 0, []
    for g in ("M1", "M2", "M3"):
        lab_all = vc.dfl_features(g, 0, our_missing_mask()).label.to_numpy()
        for s, e in vc.pm.valid_runs(lab_all):
            lab, pp, ps, pc = T.units["Metrica"][i]
            units.append((lab, (pp if keep else []) + [p[s:e] for p in nets[g]] * copies, ps, pc))
            i += 1
    T.units["Metrica"] = units
    T.units["match"] = [(a, b, (pp if keep else []) + [p[a:b] for p in nets["ours"]] * copies,
                         ps, pc) for a, b, pp, ps, pc in T.units["match"]]
    return T


def main():
    warnings.filterwarnings("ignore")
    ours, _ = features.build()
    cols = [c for c in ours.columns if not re.search(DROP, c)]
    miss = our_missing_mask()
    names = BAG + EXTRA + list(SINGLE.values())
    dead = [joblib.load(os.path.join(data.CACHE, "pff_dflpff_st_s{}.joblib".format(s)))
            for s in (0, 2, 4)]
    S = vc.Samples(plain=vc.load_models(names), state=dead)
    set_pdc(S, detector("pff", miss, cols), miss, cols, ours)

    ref = vc.evaluate(subset(S, names, BAG), FULL)
    vc.report("submission (9-model team bag)", ref, ref)
    if sys.argv[1:2] == ["tcn"]:
        stems = tuple(sys.argv[2:]) or ("dfl_tcn_all",)
        for k, w in ((3, 0.25), (9, 0.5), (27, 0.75)):
            copies = max(1, round(k / len(stems)))
            res = vc.evaluate(with_tcn(S, names, BAG, stems, copies), FULL)
            vc.report("9 models + {} network(s), weight {:.2f}".format(len(stems), w), ref, res)
            print("    official: " + official_paired(ref, res), flush=True)
        res = vc.evaluate(with_tcn(S, names, None, stems), FULL)
        vc.report("{} network(s) alone".format(len(stems)), ref, res)
        print("    official: " + official_paired(ref, res), flush=True)
        return
    for name, keep in (("15 models: + 6 plain draws", BAG + EXTRA),
                       ("12 plain draws only", vc.PLAIN + EXTRA),
                       ("6 plain draws only", vc.PLAIN)):
        res = vc.evaluate(subset(S, names, keep), FULL)
        vc.report(name, ref, res)
        print("    official: " + official_paired(ref, res), flush=True)

    one = vc.evaluate(subset(S, names, ["dfl_model.joblib"]), FULL)
    vc.report("single model: plain DFL, seed 0", one, one)
    for name, model in SINGLE.items():
        res = vc.evaluate(subset(S, names, [model]), FULL)
        vc.report(name, one, res)
        print("    official: " + official_paired(one, res), flush=True)
    res = vc.evaluate(with_tcn(S, names), FULL)
    vc.report("temporal network (TCN)", one, res)
    print("    official: " + official_paired(one, res), flush=True)


if __name__ == "__main__":
    main()
