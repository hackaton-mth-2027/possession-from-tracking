"""Turn per-frame class probabilities into possession stretches with a 3-state Viterbi decode.

States are H, A, D. Every change of state pays a cost, so a single uncertain frame cannot flip the
label: changing team needs enough accumulated evidence to pay `team_cost`, entering or leaving a
dead ball needs enough to pay `dead_cost`. Higher costs give fewer, longer possessions -- which is
what spellF1 rewards, up to the point where real short possessions get merged away.
"""

import numpy as np

H, A, D = 0, 1, 2


def transition_costs(team_cost, dead_cost):
    c = np.zeros((3, 3))
    c[H, A] = c[A, H] = team_cost
    c[H, D] = c[A, D] = c[D, H] = c[D, A] = dead_cost
    return c


def viterbi(proba, team_cost, dead_cost, eps=1e-4):
    """Most likely state path for (T, 3) probabilities. Costs are in nats."""
    logp = np.log(np.clip(proba, eps, 1.0))
    cost = transition_costs(team_cost, dead_cost)
    n = len(logp)
    back = np.zeros((n, 3), np.int8)
    score = logp[0].copy()
    for t in range(1, n):
        cand = score[:, None] - cost          # cand[i, j]: come from i, go to j
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + logp[t]
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_min_duration(proba, team_cost, dead_cost, min_len, eps=1e-4):
    """Viterbi where every stretch lasts at least min_len[state] frames (except at the edges).

    Each state is expanded into a chain of counters; a stretch can only be left from the end of
    its chain. min_len = (1, 1, 1) is the plain decode above.
    """
    min_len = [max(1, int(m)) for m in min_len]
    if max(min_len) == 1:
        return viterbi(proba, team_cost, dead_cost, eps)
    logp = np.log(np.clip(proba, eps, 1.0))
    cost = transition_costs(team_cost, dead_cost)
    n = len(logp)
    offsets = np.cumsum([0] + min_len[:-1])
    n_states = sum(min_len)
    cls = np.concatenate([[c] * m for c, m in enumerate(min_len)])
    last = offsets + np.array(min_len) - 1           # index of each class's final counter

    neg = -np.inf
    score = logp[0][cls].copy()                      # any counter may start the sequence
    enter_from = np.zeros((n, 3), np.int8)           # best previous class when entering c
    stayed = np.zeros((n, 3), bool)                  # at the final counter: stayed vs arrived
    for t in range(1, n):
        new = np.full(n_states, neg)
        ends = score[last]                           # (3,) score at the end of each chain
        cand = ends[:, None] - cost                  # from class i to class j
        np.fill_diagonal(cand, neg)
        enter_from[t] = np.argmax(cand, axis=0)
        enter = cand[enter_from[t], np.arange(3)]
        for c in range(3):
            o, m = offsets[c], min_len[c]
            if m == 1:
                stay = score[o]
                stayed[t, c] = stay >= enter[c]
                new[o] = max(stay, enter[c])
            else:
                new[o] = enter[c]
                new[o + 1:o + m - 1] = score[o:o + m - 2]
                stay, arrive = score[o + m - 1], score[o + m - 2]
                stayed[t, c] = stay >= arrive
                new[o + m - 1] = max(stay, arrive)
        score = new + logp[t][cls]

    path = np.empty(n, np.int8)
    state = int(np.argmax(score))
    for t in range(n - 1, -1, -1):
        c = cls[state]
        path[t] = c
        if t == 0:
            break
        o, m = offsets[c], min_len[c]
        j = state - o
        if j == 0 and not (m == 1 and stayed[t, c]):
            state = last[enter_from[t, c]]
        elif j == m - 1 and stayed[t, c]:
            state = state
        else:
            state = state - 1
    return path


def viterbi_events(proba, team_cost, dead_in, dead_out, eps=1e-4):
    """Viterbi with time-varying costs: dead_in[t] / dead_out[t] are the costs of entering /
    leaving a dead ball at frame t (arrays), so detected stoppage events can make those
    transitions cheap exactly where they happen."""
    logp = np.log(np.clip(proba, eps, 1.0))
    n = len(logp)
    back = np.zeros((n, 3), np.int8)
    score = logp[0].copy()
    cost = transition_costs(team_cost, 0.0)
    for t in range(1, n):
        cost[H, D] = cost[A, D] = dead_in[t]
        cost[D, H] = cost[D, A] = dead_out[t]
        cand = score[:, None] - cost
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + logp[t]
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def event_costs(n, events, base, bonus, width):
    """Per-frame cost array: `base` everywhere, `base - bonus` (at least 1) within `width` frames
    of any event."""
    near = np.zeros(n, bool)
    for f in events:
        near[max(0, f - width):f + width + 1] = True
    return np.where(near, max(1.0, base - bonus), base)


def viterbi_tv(proba, team_cost, dead_in, dead_out, eps=1e-4):
    """Viterbi with every transition cost varying over time: team_cost[t] for a change of team
    (H <-> A), dead_in[t] / dead_out[t] for entering / leaving a dead ball. Arrays of length T."""
    logp = np.log(np.clip(proba, eps, 1.0))
    n = len(logp)
    back = np.zeros((n, 3), np.int8)
    score = logp[0].copy()
    cost = np.zeros((3, 3))
    for t in range(1, n):
        cost[H, A] = cost[A, H] = team_cost[t]
        cost[H, D] = cost[A, D] = dead_in[t]
        cost[D, H] = cost[D, A] = dead_out[t]
        cand = score[:, None] - cost
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + logp[t]
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
