#!/usr/bin/env python3
"""Chapter 39: a multi-model complexity router. Entropy defined precisely, threshold calibrated on held-out data.

Run:  python3 ch39_complexity_router.py            # self-tests + experiments (CPU, numpy, ~30 s)
      python3 ch39_complexity_router.py --selftest

WHAT IS REAL: the entropy definition and its code, the routing/cost/quality accounting, the calibration procedure,
the statistics. WHAT IS INVENTED: the models and queries. There is a latent difficulty d in [0,1] per query; the small
and large "models" answer correctly with probabilities that are made-up functions of d, and the small model's token
logits are synthetic. Every number printed is a property of that synthetic world, not of any real model pair.
"""
import math
import sys
import time

import numpy as np

V = 64            # synthetic vocabulary size
K = 8             # number of generated tokens the feature averages over


# ------------------------------------------------------------------ 1. the feature, defined precisely
def token_entropy_nats(logits):
    """Shannon entropy H = -sum_v p_v ln p_v, in NATS, of softmax(logits) over the vocabulary (last axis), at temperature 1.
    Numerically stable (log-sum-exp), invariant to adding a constant to the logits, bounded by 0 <= H <= ln V."""
    z = logits - logits.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    logp = z - lse
    p = np.exp(logp)
    return -(p * logp).sum(axis=-1)


def router_feature(logits_kv):
    """THE feature: mean over the first K greedy-decoded tokens of the SMALL model's next-token entropy (nats).
    logits_kv shape (N, K, V). Requires running the small model for K tokens first, so the small model's cost is
    always paid; the large model's cost is paid only for routed queries."""
    return token_entropy_nats(logits_kv).mean(axis=-1)


# ------------------------------------------------------------------ 2. synthetic world
def p_small(d):  # probability the small model answers correctly, as a function of difficulty (INVENTED)
    return np.clip(0.97 - 0.85 * d, 0.02, 1)


def p_large(d):  # large model degrades much more slowly (INVENTED)
    return np.clip(0.98 - 0.25 * d, 0.02, 1)


def make_data(n, seed, a=2.0, b=3.0, sharp_noise=0.9):
    """n queries. d ~ Beta(a,b). Small-model logits: one token gets a boost whose size falls with difficulty, so easy
    queries give peaked (low-entropy) distributions; per-query noise in the boost makes the feature imperfect."""
    rng = np.random.default_rng(seed)
    d = rng.beta(a, b, n)
    boost = np.maximum(0.0, 7.0 * (1 - d) + sharp_noise * rng.standard_normal(n))[:, None, None]
    logits = rng.standard_normal((n, K, V)) * 1.0
    logits[:, :, 0] += boost[:, :, 0]
    H = router_feature(logits)
    length = 20 + 60 * d + 15 * rng.standard_normal(n)                 # weak baseline feature: prompt length
    u1, u2 = rng.random(n), rng.random(n)
    return dict(d=d, H=H, length=length, cs=(u1 < p_small(d)), cl=(u2 < p_large(d)))


def split(data, frac, rng):
    n = len(data["d"])
    idx = rng.permutation(n)
    a, b = idx[: int(n * frac)], idx[int(n * frac):]
    return {k: v[a] for k, v in data.items()}, {k: v[b] for k, v in data.items()}


# ------------------------------------------------------------------ 3. routing, cost, quality
def evaluate(data, score, tau):
    """Route to the large model when score > tau. Returns routed fraction, quality, and the per-query paired gap
    (router correct - large correct), which is what the confidence statements are about."""
    routed = score > tau
    correct = np.where(routed, data["cl"], data["cs"]).astype(float)
    return dict(frac=float(routed.mean()), quality=float(correct.mean()), gap=correct - data["cl"].astype(float))


def cost_ratio(frac, c_small, c_large):
    """Cost relative to always-large. The small model always runs (it produces the feature); the large model runs for
    routed queries only: (c_s + frac * c_l) / c_l."""
    return (c_small + frac * c_large) / c_large


def auc(score, positive):
    """P(score of a random positive > score of a random negative), ties count 1/2 (rank formulation)."""
    score, positive = np.asarray(score), np.asarray(positive, bool)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ 4. calibration
Z95_ONE_SIDED = 1.645


def calibrate_tau(val, score, delta=0.02, conservative=True, grid=200):
    """Choose the LARGEST threshold (fewest queries routed, so cheapest) such that quality loss versus always-large is
    within delta on the validation set.
      naive:        mean gap >= -delta
      conservative: mean gap - 1.645*SE(gap) >= -delta   (one-sided 95% lower bound on the paired gap)
    Candidate thresholds are quantiles of the validation scores. Returns tau (or -inf = route everything if none qualifies)."""
    qs = np.quantile(score, np.linspace(0, 1, grid))[::-1]       # high tau first (cheapest first)
    n = len(score)
    for tau in qs:
        g = evaluate(val, score, tau)["gap"]
        m, se = g.mean(), g.std(ddof=1) / math.sqrt(n)
        stat = m - Z95_ONE_SIDED * se if conservative else m
        if stat >= -delta:
            return float(tau)
    return -math.inf


def selftests():
    res = []

    def check(name, cond):
        res.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    z = np.zeros((3, V))
    check(f"entropy of a uniform distribution = ln V = {math.log(V):.4f} nats", np.allclose(token_entropy_nats(z), math.log(V)))
    oh = np.full((1, V), -1e4); oh[0, 5] = 0
    check("entropy of a (numerically) one-hot distribution is 0", abs(token_entropy_nats(oh)[0]) < 1e-9)
    rng = np.random.default_rng(0)
    L = rng.standard_normal((200, V)) * 3
    H = token_entropy_nats(L)
    check("0 <= H <= ln V for 200 random logit vectors", (H >= -1e-12).all() and (H <= math.log(V) + 1e-12).all())
    check("H is invariant to adding a constant to the logits", np.allclose(H, token_entropy_nats(L + 123.4)))
    two = np.full((1, V), -1e4); two[0, 0] = 0; two[0, 1] = 0
    check(f"entropy of a fair two-way split is ln 2 = {math.log(2):.4f}", abs(token_entropy_nats(two)[0] - math.log(2)) < 1e-9)
    check("H is larger for a higher-temperature (flatter) distribution", token_entropy_nats(L[:1] / 3)[0] > token_entropy_nats(L[:1] * 3)[0])

    D = make_data(20000, seed=0)
    check(f"synthetic world: correlation(H, d) = {np.corrcoef(D['H'], D['d'])[0, 1]:.3f} is positive and not 1",
          0.5 < np.corrcoef(D["H"], D["d"])[0, 1] < 0.99)
    check(f"the large model is more accurate than the small one ({D['cl'].mean():.3f} vs {D['cs'].mean():.3f})", D["cl"].mean() > D["cs"].mean())

    taus = np.quantile(D["H"], [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01])
    fr = [evaluate(D, D["H"], t)["frac"] for t in taus]
    check(f"routed fraction is nonincreasing in the threshold {[round(f, 2) for f in fr]}", all(a <= b + 1e-12 for a, b in zip(fr, fr[1:])))
    check("tau = -inf routes everything: quality = always-large exactly",
          abs(evaluate(D, D["H"], -math.inf)["quality"] - D["cl"].mean()) < 1e-12)
    check("tau = +inf routes nothing: quality = always-small exactly",
          abs(evaluate(D, D["H"], math.inf)["quality"] - D["cs"].mean()) < 1e-12)

    a_h = auc(D["H"], ~D["cs"])
    a_len = auc(D["length"], ~D["cs"])
    a_d = auc(D["d"], ~D["cs"])
    check(f"AUC for predicting small-model errors: oracle d {a_d:.3f} >= entropy {a_h:.3f} > length {a_len:.3f} > 0.5",
          a_d >= a_h > a_len > 0.5)
    check("AUC of a constant score is exactly 0.5 (ties)", abs(auc(np.zeros(100), np.arange(100) % 2 == 0) - 0.5) < 1e-12)

    cr = cost_ratio(0.3, 1.0, 15.0)
    check(f"cost ratio by hand: (1 + 0.3*15)/15 = {5.5 / 15:.4f}; function gives {cr:.4f}", abs(cr - 5.5 / 15) < 1e-12)
    check("with frac=1 the router costs more than always-large: (c_s + c_l)/c_l > 1", cost_ratio(1.0, 1.0, 15.0) > 1)

    rng = np.random.default_rng(1)
    val, test = split(make_data(30000, seed=2), 0.5, rng)
    tau = calibrate_tau(val, val["H"], delta=0.02, conservative=True)
    gv = evaluate(val, val["H"], tau)["gap"].mean()
    check(f"calibration meets its own criterion on validation: gap {gv:+.4f} >= -0.02 (tau={tau:.3f})", gv >= -0.02 - 1e-9)
    check("conservative threshold is no higher than the naive one (it routes at least as many queries)",
          calibrate_tau(val, val["H"], conservative=True) <= calibrate_tau(val, val["H"], conservative=False) + 1e-12)
    return all(res)


def experiments():
    D = make_data(60000, seed=0)
    c_s, c_l = 1.0, 15.0
    always_l, always_s = D["cl"].mean(), D["cs"].mean()

    print(f"\n== 1. The feature: entropy (nats) vs difficulty. always-small quality {always_s:.3f}, always-large {always_l:.3f} ==")
    print(f"  corr(H, d) = {np.corrcoef(D['H'], D['d'])[0, 1]:.3f};  H range [{D['H'].min():.2f}, {D['H'].max():.2f}], ln V = {math.log(V):.2f}")
    print(f"  AUC for 'small model wrong':  oracle d {auc(D['d'], ~D['cs']):.3f} | entropy {auc(D['H'], ~D['cs']):.3f} | prompt length {auc(D['length'], ~D['cs']):.3f}")

    print("\n== 2. Frontier: routed fraction -> quality and cost (c_small=1, c_large=15), by routing signal ==")
    print(f"  {'route frac':>10s} | {'oracle d':>9s} {'entropy':>9s} {'length':>9s} {'random':>9s}   quality;  cost ratio = {cost_ratio(0.3, c_s, c_l):.3f} at 0.30")
    rng = np.random.default_rng(5)
    rnd = rng.random(len(D["d"]))
    for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        row = []
        for name in ("d", "H", "length"):
            tau = np.quantile(D[name], 1 - f) if f < 1 else -math.inf
            row.append(evaluate(D, D[name], tau)["quality"])
        tau = np.quantile(rnd, 1 - f) if f < 1 else -math.inf
        row.append(evaluate(D, rnd, tau)["quality"])
        print(f"  {f:10.2f} | " + " ".join(f"{q:9.3f}" for q in row) + f"   cost {cost_ratio(f, c_s, c_l):.3f}")

    print("\n== 3. Calibration: naive vs conservative, 300 random 50/50 splits of 4000 queries (delta = 0.02) ==")
    out = {"naive": [], "conservative": []}
    for s in range(300):
        d = make_data(4000, seed=1000 + s)
        v, t = split(d, 0.5, np.random.default_rng(s))
        for mode in out:
            tau = calibrate_tau(v, v["H"], 0.02, conservative=(mode == "conservative"))
            e = evaluate(t, t["H"], tau)
            out[mode].append((e["gap"].mean(), e["frac"]))
    print(f"  {'method':14s} {'test gap mean':>13s} {'P(test gap < -0.02)':>20s} {'routed frac':>12s} {'cost ratio':>11s}")
    for mode, arr in out.items():
        a = np.array(arr)
        print(f"  {mode:14s} {a[:, 0].mean():13.4f} {np.mean(a[:, 0] < -0.02):20.3f} {a[:, 1].mean():12.3f} {cost_ratio(a[:, 1].mean(), c_s, c_l):11.3f}")

    print("\n== 4. Distribution shift: calibrate on Beta(2,3) difficulty, deploy on harder traffic ==")
    v = make_data(30000, seed=7)
    tau = calibrate_tau(v, v["H"], 0.02, True)
    print(f"  calibrated tau = {tau:.3f} nats")
    for name, (a, b) in (("same traffic Beta(2,3)", (2, 3)), ("harder Beta(3,2)", (3, 2)), ("much harder Beta(5,2)", (5, 2)), ("easier Beta(2,5)", (2, 5))):
        t = make_data(30000, seed=8, a=a, b=b)
        e = evaluate(t, t["H"], tau)
        print(f"  {name:24s}: routed {e['frac']:.3f}  quality {e['quality']:.3f}  always-large {t['cl'].mean():.3f}  gap {e['gap'].mean():+.4f}")

    print("\n== 5. When does the router pay? cost ratio at the calibrated routed fraction, by large/small price ratio ==")
    tau = calibrate_tau(v, v["H"], 0.02, True)
    f = evaluate(D, D["H"], tau)["frac"]
    print(f"  calibrated routed fraction on the 60k set: {f:.3f}; break-even price ratio c_l/c_s = 1/(1-f) = {1 / (1 - f):.2f}")
    for r in (2, 5, 10, 15, 30, 100):
        print(f"  c_l/c_s = {r:4d}: cost ratio {cost_ratio(f, 1.0, float(r)):.3f}  ({'saves' if cost_ratio(f, 1.0, float(r)) < 1 else 'COSTS MORE than always-large'})")


if __name__ == "__main__":
    t0 = time.time()
    print("self-tests:")
    ok = selftests()
    if "--selftest" not in sys.argv:
        experiments()
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}   ({time.time() - t0:.0f} s)")
    sys.exit(0 if ok else 1)
