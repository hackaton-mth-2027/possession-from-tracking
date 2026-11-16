"""A temporal model of "who has the ball", trained on the 7 DFL matches, applied to ours.

    .venv/bin/python scripts/dfl_tcn.py [--epochs 6] [--out dfl_tcn_proba.npy]
        [--max-dil 64] [--channels 64] [--noise plain|st] [--extra] [--weighted] [--touch]
        [--metrica M1,M2 --met-weight 3] [--pff --pff-weight 0.2] [--dfl-target lite]
        [--arch tcn|gru] [--passf] [--mirror]

The gradient-boosting model of scripts/dfl_pretrain.py sees one frame at a time (with a few
rolling means). Possessions of 1-4 s are decided by what happens around the frame: who is near the
ball when it lands, who moves with it next. This model looks at +-5 s:

  per-frame features (same columns as dfl_pretrain.py, standardised, missing -> 0)
  -> 1x1 conv to 64 channels
  -> 7 residual blocks of dilated convolutions (kernel 3, dilation 1..64, centred: past and future)
  -> 3 logits per frame (home, away, dead)

Trained on the noisy DFL copies (cache/dfl_feat_*.pkl, built by dfl_pretrain.py) on random
windows of 512 frames, one match held out to report accuracy. CPU, fixed seeds, deterministic.
Writes cache/<out> (N_FRAMES, 3) for our match: use with DFL=1 DFL_PROBA=<out>.
"""

import argparse
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from possession import data, dfl, features  # noqa: E402
from dfl_pretrain import (DROP, dfl_features, extra_features, our_missing_mask,  # noqa: E402
                          pass_features)

WIN = 512
DILATIONS = (1, 2, 4, 8, 16, 32, 64)
HOLD_OUT = "J03WR9"


class Block(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.norm = nn.GroupNorm(8, ch)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return x + self.drop(torch.relu(self.norm(self.conv(x))))


class TCN(nn.Module):
    def __init__(self, n_in, ch=64, dilations=DILATIONS):
        super().__init__()
        self.inp = nn.Conv1d(n_in, ch, 1)
        self.blocks = nn.Sequential(*[Block(ch, d) for d in dilations])
        self.out = nn.Conv1d(ch, 3, 1)

    def forward(self, x):                     # x: (B, C, T)
        return self.out(self.blocks(torch.relu(self.inp(x))))


class GRUNet(nn.Module):
    """The same inputs and outputs as TCN, through a 2-layer bidirectional GRU."""

    def __init__(self, n_in, ch=64):
        super().__init__()
        self.inp = nn.Linear(n_in, ch)
        self.gru = nn.GRU(ch, ch, num_layers=2, batch_first=True, bidirectional=True,
                          dropout=0.1)
        self.out = nn.Linear(2 * ch, 3)

    def forward(self, x):                     # x: (B, C, T)
        h, _ = self.gru(torch.relu(self.inp(x.transpose(1, 2))))
        return self.out(h).transpose(1, 2)


def make_net(arch, n_in, ch, dilations):
    return GRUNet(n_in, ch) if arch == "gru" else TCN(n_in, ch, dilations)


def load_net(stem):
    """(network, columns, mean, std) saved by a training run with --out <stem>.npy."""
    ck = torch.load(os.path.join(data.CACHE, stem + ".pt"), weights_only=False)
    net = make_net(ck.get("arch", "tcn"), len(ck["cols"]), ck["channels"], ck["dilations"])
    net.load_state_dict(ck["state"])
    return net, ck["cols"], ck["mean"], ck["std"]


def possession_weights(lab):
    """Per frame: min(4, max(0.25, 4 s / length of its possession)); dead ball 1. spellF1 counts
    possessions, not frames, so short possessions should count as much as long ones."""
    w = np.ones(len(lab), np.float32)
    edges = np.flatnonzero(np.diff(lab)) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(lab)]):
        if lab[a] in (0, 1):
            w[a:b] = min(4.0, max(0.25, 4.0 * data.FPS / (b - a)))
    return w


def prepare(df, cols, mean, std):
    X = (df[cols].to_numpy(np.float32) - mean) / std
    return np.nan_to_num(np.clip(X, -6, 6)).astype(np.float32)


def predict(net, X, chunk=4096, pad=256):
    """Probabilities for a whole sequence, in overlapping chunks (pad >= receptive field)."""
    net.eval()
    n = len(X)
    out = np.zeros((n, 3), np.float32)
    with torch.no_grad():
        for s in range(0, n, chunk):
            a, b = max(0, s - pad), min(n, s + chunk + pad)
            x = torch.from_numpy(X[a:b].T[None])
            p = torch.softmax(net(x), dim=1)[0].numpy().T
            out[s:min(n, s + chunk)] = p[s - a:s - a + min(chunk, n - s)]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--steps", type=int, default=400, help="batches per epoch")
    p.add_argument("--copies", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all", action="store_true", help="also train on the held-out match")
    p.add_argument("--out", default="dfl_tcn_proba.npy")
    p.add_argument("--max-dil", type=int, default=64, help="largest dilation (receptive field)")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--noise", default="plain", choices=("plain", "st", "cal", "stcal"),
                   help="ball gaps at random places or by game state (st); cal: the calibrated "
                        "noise of dfl_pretrain.CALIBRATED (smooth ball, as ours)")
    p.add_argument("--extra", action="store_true",
                   help="add possession/features_extra.py (interception, carrier, goalkeeper)")
    p.add_argument("--weighted", action="store_true",
                   help="loss weighted by possession length (possession_weights)")
    p.add_argument("--touch", action="store_true",
                   help="keep the touch features (features.touch_features), left out by default")
    p.add_argument("--metrica", default="",
                   help="Metrica matches to train on as well, e.g. M1,M2 (labels: METRICA_LABELS)")
    p.add_argument("--met-weight", type=float, default=1.0,
                   help="how much more often a Metrica window is drawn than its length says")
    p.add_argument("--arch", default="tcn", choices=("tcn", "gru"))
    p.add_argument("--mirror", action="store_true",
                   help="also train on every DFL copy with home and away swapped")
    p.add_argument("--passf", action="store_true",
                   help="add the pass features (dfl_pretrain.pass_features, from clean tracking)")
    p.add_argument("--pass-keep", type=float, default=1.0,
                   help="share of the training matches' detected flights kept")
    p.add_argument("--dfl-target", default="touch",
                   help="DFL label to learn (possession/dfl.labels_variant): touch (DFL's flag), "
                        "lite / strict (the challenge's controlled-possession rule)")
    p.add_argument("--pff", action="store_true",
                   help="also train on the 64 PFF World Cup matches (labels: PFF_LABELS)")
    p.add_argument("--pff-weight", type=float, default=0.2,
                   help="how often a PFF window is drawn, relative to its length")
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    rng = np.random.default_rng(args.seed)

    miss = our_missing_mask()
    ours, _ = features.build()
    drop = DROP.replace(r"(^|vis_)touch_|", "") if args.touch else DROP
    cols = [c for c in ours.columns if not re.search(drop, c)]
    pools = None
    if args.noise in ("st", "stcal"):
        from dfl_pretrain import state_pools
        pools = state_pools()
    cal = args.noise in ("cal", "stcal")
    targets = {}

    def frames(mid, s, pools=None):
        d = dfl_features(mid, s, miss, pools, calibrated=cal and not mid.startswith("M"))
        if args.dfl_target != "touch" and mid in dfl.MATCHES:
            if mid not in targets:
                targets[mid] = dfl.labels_variant(mid, dfl.load(mid), args.dfl_target)
            d = d.copy()
            d["label"] = targets[mid]
        if args.extra:
            x = extra_features(mid, s, miss, pools, calibrated=cal and not mid.startswith("M"))
            d = pd.concat([d, x], axis=1)
        if args.passf:
            d = pd.concat([d, pass_features(mid, args.pass_keep)], axis=1)
        return d

    if args.extra:
        ours = pd.concat([ours, extra_features("ours", 0, miss)], axis=1)
        cols += list(extra_features("ours", 0, miss).columns)
    if args.passf:
        ours = pd.concat([ours, pass_features("ours")], axis=1)
        cols += list(pass_features("ours").columns)
    sets = {mid: [frames(mid, s, pools) for s in range(args.copies)] for mid in dfl.MATCHES}
    if args.mirror:
        for mid in dfl.MATCHES:
            sets[mid] += [dfl_features(mid, s, miss, pools, mirror=True)
                          for s in range(args.copies)]
    dilations = tuple(d for d in (1, 2, 4, 8, 16, 32, 64, 128, 256) if d <= args.max_dil)
    train_ids = [m for m in dfl.MATCHES if args.all or m != HOLD_OUT]
    allX = pd.concat([d[cols] for m in train_ids for d in sets[m]])
    mean = np.nanmean(allX.to_numpy(np.float32), axis=0)
    std = np.nanstd(allX.to_numpy(np.float32), axis=0) + 1e-6
    del allX
    seqs, draw = [], []
    for m in train_ids:
        for d in sets[m]:
            seqs.append((prepare(d, cols, mean, std), d.label.to_numpy().astype(np.int64)))
            draw.append(1.0)
    for g in [m for m in args.metrica.split(",") if m]:
        for s in range(args.copies):
            d = frames(g, s)
            seqs.append((prepare(d, cols, mean, std), d.label.to_numpy().astype(np.int64)))
            draw.append(args.met_weight)
    if args.pff:
        from possession import pff
        for gid in pff.games():
            d = frames("P" + gid, 0)
            seqs.append((prepare(d, cols, mean, std), d.label.to_numpy().astype(np.int64)))
            draw.append(args.pff_weight)
    lengths = np.array([len(x) for x, _ in seqs], float) * np.array(draw)
    weights = [possession_weights(y) for _, y in seqs] if args.weighted else None

    net = make_net(args.arch, len(cols), args.channels, dilations)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3,
                                                total_steps=args.epochs * args.steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=3)
    for ep in range(args.epochs):
        net.train()
        t0, tot = time.time(), 0.0
        for _ in range(args.steps):
            xb, yb, wb = [], [], []
            for _ in range(32):
                k = rng.choice(len(seqs), p=lengths / lengths.sum())
                x, y = seqs[k]
                s = rng.integers(0, len(x) - WIN)
                xb.append(x[s:s + WIN].T)
                yb.append(y[s:s + WIN])
                if weights is not None:
                    wb.append(weights[k][s:s + WIN])
            xb = torch.from_numpy(np.stack(xb))
            yb = torch.from_numpy(np.stack(yb))
            if weights is None:
                loss = loss_fn(net(xb), yb)
            else:
                per = nn.functional.cross_entropy(net(xb), yb, ignore_index=3, reduction="none")
                wt = torch.from_numpy(np.stack(wb)) * (yb != 3)
                loss = (per * wt).sum() / wt.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        msg = "epoch {} loss {:.3f} ({:.0f} s)".format(ep + 1, tot / args.steps, time.time() - t0)
        if not args.all:
            d = sets[HOLD_OUT][0]
            pr = predict(net, prepare(d, cols, mean, std))
            y = d.label.to_numpy()
            live = y < 2
            msg += "  held-out {}: team {:.3f}, dead/alive {:.3f}".format(
                HOLD_OUT, np.mean(np.argmax(pr[live, :2], 1) == y[live]),
                np.mean((pr[y < 3, 2] > 0.5) == (y[y < 3] == 2)))
        print(msg, flush=True)

    proba = predict(net, prepare(ours, cols, mean, std))
    np.save(os.path.join(data.CACHE, args.out), proba)
    torch.save({"state": net.state_dict(), "cols": cols, "mean": mean, "std": std,
                "channels": args.channels, "dilations": dilations, "arch": args.arch},
               os.path.join(data.CACHE, os.path.splitext(args.out)[0] + ".pt"))
    print("wrote cache/" + args.out)
    for g in ("M1", "M2", "M3"):           # the Metrica validation games (scripts/retest.py)
        pg = predict(net, prepare(frames(g, 0), cols, mean, std))
        np.save(os.path.join(data.CACHE, os.path.splitext(args.out)[0] + "_" + g + ".npy"), pg)

    sys.path.insert(0, os.path.join(ROOT, "tools"))
    sys.path.insert(0, os.path.join(ROOT, "repo", "evaluation"))
    import possession_metrics as pm
    from dfl_pretrain import team_by_true_possession
    from error_anatomy import truth
    gt = truth()
    live = np.isin(gt, [pm.H, pm.A])
    print("On the official window (never trained on): frame team accuracy {:.3f}".format(
        np.mean(np.argmax(proba[live, :2], 1) == gt[live])))
    print(team_by_true_possession(gt, proba).to_string())


if __name__ == "__main__":
    main()
