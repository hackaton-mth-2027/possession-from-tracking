"""Re-validate the submission's choices with allowed sources only: the Metrica validation set
(3 open Metrica matches, other fixtures), the official public 5 minutes, and label-free statistics
of the full match. No label of the competition match is trained on (RULES.md).

    .venv/bin/python scripts/validate_clean.py decoder      # step, team cost, bonus, dead cost
    .venv/bin/python scripts/validate_clean.py low          # lower costs than the first grid
    .venv/bin/python scripts/validate_clean.py final        # bonus ratio 1, per-game check
    .venv/bin/python scripts/validate_clean.py step         # step length at the chosen costs
    .venv/bin/python scripts/validate_clean.py calib        # team temperature, label-free prior
    .venv/bin/python scripts/validate_clean.py models       # which models give team / dead ball

Every configuration is decoded like the submission (per half, second half from the kickoff) and
scored with the official spellF1; Metrica differences are paired over its 1,117 possessions
(90% interval) against the current submission's configuration. The full match's possession rate is
shown next to the public window's (4.2 per live minute: the only published figure).
"""

import os
import sys
import warnings

import joblib
import numpy as np
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
from possession import data, decode, features  # noqa: E402
import possession_metrics as pm  # noqa: E402
from dfl_pretrain import DROP, dfl_features, our_missing_mask  # noqa: E402
from dfl_change import inputs  # noqa: E402
from compare import flags_and_preds, f1  # noqa: E402

import re  # noqa: E402

PLAIN = ["dfl_model.joblib"] + ["dfl_model_s{}.joblib".format(s) for s in (2, 4, 6, 8, 10)]
STATE = ["dfl_model_st.joblib", "dfl_model_st_s2.joblib", "dfl_model_st_s4.joblib"]
CURRENT = dict(step=5, team=48.0, bonus=24.0, dead=128.0, team_models="all", dead_models="st")


def load_models(names):
    return [joblib.load(os.path.join(data.CACHE, n)) for n in names]


class Samples:
    """Frame and change probabilities on the Metrica units and on our match's halves; by default
    from the submission's models, or from the `plain` / `state` models and change detector
    `change` ((model, columns)) given."""

    def __init__(self, plain=None, state=None, change=None):
        warnings.filterwarnings("ignore")
        ours, _ = features.build()
        self.cols = [c for c in ours.columns if not re.search(DROP, c)]
        det, dcols = change or joblib.load(os.path.join(data.CACHE, "dfl_change_touch.joblib"))
        plain = plain if plain is not None else load_models(PLAIN)
        state = state if state is not None else load_models(STATE)
        miss = our_missing_mask()
        self.units = {"Metrica": [], "match": []}
        self.game = []
        for g in ("M1", "M2", "M3"):
            fr = dfl_features(g, 0, miss)
            lab = fr.label.to_numpy().astype(np.int8)
            X = fr[self.cols].to_numpy(np.float32)
            pp = [m.predict_proba(X) for m in plain]
            ps = [m.predict_proba(X) for m in state]
            pc = det.predict_proba(inputs(fr, dcols))[:, 1]
            for s, e in pm.valid_runs(lab):
                self.units["Metrica"].append((lab[s:e], [p[s:e] for p in pp],
                                              [p[s:e] for p in ps], pc[s:e]))
                self.game.append(g)
        X = ours[self.cols].to_numpy(np.float32)
        pp = [m.predict_proba(X) for m in plain]
        ps = [m.predict_proba(X) for m in state]
        pc = det.predict_proba(inputs(ours, dcols))[:, 1]
        for half, a, b in data.halves():
            if half == 2:
                a = data.KICKOFF_2ND
            self.units["match"].append((a, b + 1, [p[a:b + 1] for p in pp],
                                        [p[a:b + 1] for p in ps], pc[a:b + 1]))
        self.official = pm.label_codes(data.load_public_gt())


def mix(pp, ps, team_models, dead_models, temp=1.0, home_prior=None):
    """Frame probabilities: dead ball from `dead_models`, home share from `team_models`. `temp`
    sharpens (< 1) or softens (> 1) the home share; `home_prior` re-weights it from a balanced
    prior to that home share of possession (Saerens et al. 2002)."""
    team = {"all": pp + ps, "plain": pp, "st": ps}[team_models]
    dead = {"all": pp + ps, "plain": pp, "st": ps}[dead_models]
    d = np.mean([p[:, 2] for p in dead], axis=0)
    home = np.mean([p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-9) for p in team], axis=0)
    if temp != 1.0:
        a, b = home ** (1 / temp), (1 - home) ** (1 / temp)
        home = a / np.maximum(a + b, 1e-300)
    if home_prior is not None:
        a, b = home * home_prior, (1 - home) * (1 - home_prior)
        home = a / np.maximum(a + b, 1e-300)
    return np.column_stack([(1 - d) * home, (1 - d) * (1 - home), d])


def em_home_prior(pieces, iters=50):
    """Home share of possession of one match, from its frames' probabilities alone: EM on the
    prior (live frames weighted by 1 - P(dead)), starting from the balanced training prior."""
    proba = np.vstack(pieces)
    home = proba[:, 0] / np.maximum(proba[:, 0] + proba[:, 1], 1e-9)
    w = proba[:, 0] + proba[:, 1]
    pi = 0.5
    for _ in range(iters):
        a, b = home * pi, (1 - home) * (1 - pi)
        pi = np.sum(w * a / np.maximum(a + b, 1e-300)) / w.sum()
    return pi


def decode_unit(proba, pc, cfg, pdc=None):
    """`pdc`: probability of a dead-ball boundary (stoppage or restart) per frame; with
    cfg["dead_bonus"] the dead-ball cost becomes max(4, dead - dead_bonus * pdc)."""
    R = cfg["step"]
    n = len(proba) // R * R
    lp = np.log(np.clip(proba[:n], 1e-6, 1.0)).reshape(-1, R, 3).sum(1)
    pcs = pc[:n].reshape(-1, R).max(1)
    tc = np.maximum(4.0, cfg["team"] - cfg["bonus"] * pcs)
    dd = np.full(len(pcs), cfg["dead"])
    if pdc is not None and cfg.get("dead_bonus"):
        dd = np.maximum(4.0, dd - cfg["dead_bonus"] * pdc[:n].reshape(-1, R).max(1))
    st = decode.viterbi_tv(np.exp(lp - np.logaddexp.reduce(lp, 1, keepdims=True)), tc, dd, dd,
                           eps=1e-300)
    out = np.empty(len(proba), np.int8)
    out[:n] = np.repeat(st, R)
    out[n:] = st[-1]
    return out


def unit_proba(S, key, i, pp, ps, cfg):
    prior = None
    if cfg.get("prior"):
        group = S.game[i] if key == "Metrica" else "match"
        prior = S.priors[(group, cfg["team_models"], cfg["dead_models"], cfg.get("temp", 1.0))]
    return mix(pp, ps, cfg["team_models"], cfg["dead_models"], cfg.get("temp", 1.0), prior)


def set_priors(S, cfg):
    """EM home prior per Metrica match and for our match, for this configuration."""
    if not hasattr(S, "priors"):
        S.priors = {}
    groups = {}
    for i, (_, pp, ps, _) in enumerate(S.units["Metrica"]):
        groups.setdefault(S.game[i], []).append((pp, ps))
    groups["match"] = [(pp, ps) for _, _, pp, ps, _ in S.units["match"]]
    for g, members in groups.items():
        key = (g, cfg["team_models"], cfg["dead_models"], cfg.get("temp", 1.0))
        S.priors[key] = em_home_prior([mix(pp, ps, *key[1:]) for pp, ps in members])


def evaluate(S, cfg):
    if cfg.get("prior"):
        set_priors(S, cfg)
    fl, npd, labs, preds = [], [], [], []
    pdc = getattr(S, "pdc", {"Metrica": [None] * len(S.units["Metrica"]),
                             "match": [None] * len(S.units["match"])})
    for i, (lab, pp, ps, pc) in enumerate(S.units["Metrica"]):
        pred = decode_unit(unit_proba(S, "Metrica", i, pp, ps, cfg), pc, cfg, pdc["Metrica"][i])
        f, n = flags_and_preds(lab, pred, [(0, len(lab))])
        fl.append(f)
        npd.append(n)
        labs.append(lab)
        preds.append(pred)
    macro = f1_score(np.concatenate(labs), np.concatenate(preds), labels=[pm.H, pm.A, pm.D],
                     average="macro")
    full = np.full(data.N_FRAMES, pm.X, np.int8)
    full[110808:data.KICKOFF_2ND] = pm.D
    for j, (a, b, pp, ps, pc) in enumerate(S.units["match"]):
        full[a:b] = decode_unit(unit_proba(S, "match", None, pp, ps, cfg), pc, cfg,
                                pdc["match"][j])
    off = pm.evaluate(S.official, np.where(S.official == pm.X, pm.X, full))
    graded = full != pm.X
    graded[110808:data.KICKOFF_2ND] = False
    runs = [(a, b + 1) for _, a, b in data.halves()]
    n_poss = len(pm.possession_spells(full, runs))
    live = np.isin(full, [pm.H, pm.A]).sum() / data.FPS / 60
    return ((np.hstack(fl), np.concatenate(npd)), off["spell_f1_mean"], n_poss, n_poss / live,
            macro, off["macro_f1"], fl, np.mean(full[graded] == pm.D),
            np.mean(np.concatenate(preds) == pm.D),
            flags_and_preds(S.official, np.where(S.official == pm.X, pm.X, full),
                            pm.valid_runs(S.official)), full)


def paired(ref, var):
    fa, na = ref
    fb, nb = var
    n = fa.shape[1]
    r = np.random.RandomState(0)
    d = np.array([f1(fb, nb, k) - f1(fa, na, k) for k in (r.randint(0, n, n)
                                                         for _ in range(1000))])
    allk = np.arange(n)
    return f1(fb, nb, allk), f1(fb, nb, allk) - f1(fa, na, allk), np.percentile(d, [5, 95])


def report(name, ref, res):
    sc, diff, (lo, hi) = paired(ref[0], res[0])
    print("{:<44s} Metrica {:.3f} {:+.3f} [{:+.3f}, {:+.3f}] macro {:.3f} D {:.0%} | official "
          "{:.3f} macro {:.3f} | match {} poss, {:.2f}/live min, D {:.0%}".format(
              name, sc, diff, lo, hi, res[4], res[8], res[1], res[5], res[2], res[3], res[7]),
          flush=True)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "decoder"
    S = Samples()
    ref = evaluate(S, CURRENT)
    report("CURRENT (step 5, 48/24, dead 128, team all, dead st)", ref, ref)
    if what == "decoder":
        for step in (1, 5):
            for team in (24.0, 32.0, 40.0, 48.0, 64.0):
                for ratio in (0.5, 0.67):
                    for dead in (64.0, 128.0, 192.0):
                        cfg = dict(CURRENT, step=step, team=team, bonus=round(team * ratio, 1),
                                   dead=dead)
                        report("step {} team {:.0f} bonus {:.0f} dead {:.0f}".format(
                            step, team, cfg["bonus"], dead), ref, evaluate(S, cfg))
    elif what == "calib":
        # team temperature and label-free home prior, at the submission's costs
        base = dict(CURRENT, step=1, team=28.0, bonus=23.0, dead=64.0)
        ref1 = evaluate(S, base)
        report("submission (28/23/64, frame by frame)", ref1, ref1)
        for temp in (0.5, 0.75, 1.5, 2.0):
            report("team temperature {:g}".format(temp), ref1, evaluate(S, dict(base, temp=temp)))
        for temp in (0.75, 1.0):
            res = evaluate(S, dict(base, temp=temp, prior=True))
            pri = {k[0]: v for k, v in S.priors.items() if k[3] == temp}
            report("EM home prior, temperature {:g}".format(temp), ref1, res)
            print("    estimated home shares: " + ", ".join(
                "{} {:.2f}".format(k, v) for k, v in sorted(pri.items())), flush=True)
    elif what == "step":
        # steps of 1 / 3 / 5 / 10 frames at the chosen costs, paired against steps of 5
        base = dict(CURRENT, team=28.0, bonus=23.0, dead=64.0)
        ref5 = evaluate(S, base)
        for team, bonus in ((28.0, 23.0), (24.0, 20.0), (20.0, 17.0)):
            for step in (1, 3, 5, 10):
                cfg = dict(base, team=team, bonus=bonus, step=step)
                report("team {:.0f} bonus {:.0f} dead 64 step {} (vs 28/23/64 step 5)".format(
                    team, bonus, step), ref5, evaluate(S, cfg))
    elif what == "final":
        # bonus ratio up to 1 (cost = max(4, team * (1 - p_change))), then per Metrica game
        for team in (20.0, 24.0, 28.0, 32.0):
            for ratio in (0.83, 1.0):
                for dead in (32.0, 48.0, 64.0):
                    cfg = dict(CURRENT, team=team, bonus=round(team * ratio, 1), dead=dead)
                    report("step 5 team {:.0f} bonus {:.0f} dead {:.0f}".format(
                        team, cfg["bonus"], dead), ref, evaluate(S, cfg))
        for name, cfg in (("team 24 bonus 20 dead 64", dict(CURRENT, team=24.0, bonus=20.0,
                                                            dead=64.0)),
                          ("team 28 bonus 23 dead 64", dict(CURRENT, team=28.0, bonus=23.0,
                                                            dead=64.0)),
                          ("team 24 bonus 20 dead 32", dict(CURRENT, team=24.0, bonus=20.0,
                                                            dead=32.0))):
            res = evaluate(S, cfg)
            cells = []
            for g in ("M1", "M2", "M3"):
                keep = np.flatnonzero(np.repeat(S.game, [f.shape[1] for f in res[6]]) == g)
                fa, na = ref[0]
                fb, nb = res[0]
                cells.append("{} {:+.3f} ({} poss)".format(g, f1(fb, nb, keep) - f1(fa, na, keep),
                                                           len(keep)))
            print("per game, {} vs CURRENT: {}".format(name, " | ".join(cells)), flush=True)
    elif what == "low":
        # below the first grid's edge (team 24, dead 64), steps of 5 frames
        for team in (12.0, 16.0, 20.0, 24.0, 28.0):
            for ratio in (0.5, 0.67, 0.83):
                for dead in (24.0, 32.0, 48.0, 64.0, 96.0):
                    cfg = dict(CURRENT, team=team, bonus=round(team * ratio, 1), dead=dead)
                    report("step 5 team {:.0f} bonus {:.0f} dead {:.0f}".format(
                        team, cfg["bonus"], dead), ref, evaluate(S, cfg))
    else:
        for tm in ("all", "plain", "st"):
            for dm in ("st", "all", "plain"):
                cfg = dict(CURRENT, team_models=tm, dead_models=dm)
                report("team from {}, dead from {}".format(tm, dm), ref, evaluate(S, cfg))


if __name__ == "__main__":
    main()
