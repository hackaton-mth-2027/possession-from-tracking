"""Is the score limited by the model or by the information in our tracking?

    .venv/bin/python scripts/model_vs_information.py

On the 7 DFL matches, where the truth is known: each match predicted by a gradient-boosting model
trained on the other six (leave one match out), decoded with the submission's Viterbi, scored
with the official spellF1 against DFL's possession (every touch, and opponent spells < 2 s removed,
which is closer to our ground truth's rate). Two worlds, same model and decoder:
  clean   DFL's own tracking: exact positions, ball always seen
  noisy   made to look like ours (possession/dfl.add_noise: camera projection of airborne balls,
          ~0.5 m jitter, ball missing 41% of the time, player fragments) -- the probabilities of
          scripts/dfl_decoder_costs.py (cache/dfl_lomo_<match>.npy)
If clean >> noisy, our tracking's noise is what limits (better tracking / vision is the lever).
If clean is low too, the model (or the definition) is what limits (better models are the lever).
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
from possession import data, decode, dfl, features  # noqa: E402
import possession_metrics as pm  # noqa: E402
from dfl_pretrain import DROP, STEP, make_model  # noqa: E402

COSTS = ((64.0, 128.0), (16.0, 64.0))
TARGETS = ("touch", "sandwich2")


def clean_features(mid):
    path = os.path.join(data.CACHE, "dfl_feat_{}_clean.pkl".format(mid))
    if os.path.exists(path):
        return pd.read_pickle(path)
    m = dfl.load(mid)
    df, _ = features.build(match=m)
    df["label"] = m["labels"]
    df.to_pickle(path)
    return df


def lomo_clean(cols):
    sets = {mid: clean_features(mid) for mid in dfl.MATCHES}
    out = {}
    for mid in dfl.MATCHES:
        path = os.path.join(data.CACHE, "dfl_lomo_clean_{}.npy".format(mid))
        if os.path.exists(path):
            out[mid] = np.load(path)
            continue
        tr = pd.concat([sets[m].iloc[::STEP] for m in dfl.MATCHES if m != mid])
        tr = tr[tr.label < 3]
        model = make_model().fit(tr[cols].to_numpy(np.float32), tr.label.to_numpy())
        out[mid] = model.predict_proba(sets[mid][cols].to_numpy(np.float32)).astype(np.float32)
        np.save(path, out[mid])
        print("  clean, held-out", mid, flush=True)
    return out


def score(prob, lab, tc, dc):
    runs = pm.valid_runs(lab)
    pred = np.full(len(lab), pm.X, np.int8)
    for s, e in runs:
        pred[s:e] = decode.viterbi(prob[s:e].astype(np.float64), tc, dc)
    live = np.isin(lab, (0, 1))
    team = np.mean(np.argmax(prob[live, :2], 1) == lab[live])
    return pm.spell_scores(lab, pred, runs)["mean_f1"], team


def main():
    warnings.filterwarnings("ignore")
    ours, _ = features.build()
    cols = [c for c in ours.columns if not re.search(DROP, c)]
    clean = lomo_clean(cols)
    noisy = {mid: np.load(os.path.join(data.CACHE, "dfl_lomo_{}.npy".format(mid)))
             for mid in dfl.MATCHES}
    rows = []
    for mid in dfl.MATCHES:
        m = dfl.load(mid)
        for target in TARGETS:
            lab = dfl.labels_variant(mid, m, target).astype(np.int8)
            for tc, dc in COSTS:
                for world, prob in (("clean", clean[mid]), ("noisy", noisy[mid])):
                    f1, team = score(prob, lab, tc, dc)
                    rows.append(dict(match=mid, target=target, costs="{:.0f}/{:.0f}".format(
                        tc, dc), world=world, spellF1=f1, frame_team=team))
    df = pd.DataFrame(rows)
    print(df.groupby(["target", "costs", "world"])[["spellF1", "frame_team"]].mean().round(3)
          .unstack("world").to_string())


if __name__ == "__main__":
    main()
