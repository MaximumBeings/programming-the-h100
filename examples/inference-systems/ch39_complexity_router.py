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
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 0 or logits.shape[-1] == 0:
        raise ValueError("logits must have a nonempty vocabulary axis")
    if np.isnan(logits).any() or np.isposinf(logits).any():
        raise ValueError("logits may contain finite values or negative-infinity masks only")
    maximum = logits.max(axis=-1, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError("each row needs at least one unmasked token")
    z = logits - maximum
    lse = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    logp = z - lse
    p = np.exp(logp)
    # lim(p*log(p), p->0) = 0, including exactly masked logits.
    terms = np.zeros_like(p)
    np.multiply(p, logp, out=terms, where=p > 0)
    return -terms.sum(axis=-1)


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
    n = len(next(iter(data.values())))
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


def cost_ratio(frac, c_small, c_large, prefix_frac=1.0):
    """Cost relative to always-large. The feature needs only the first K tokens of the small model's answer.
      unrouted queries: the small model finishes its answer              -> c_s
      routed queries:   the K-token prefix is wasted, then the large model -> prefix_frac * c_s + c_l
    prefix_frac = cost of the K-token prefix / cost of a full small-model answer, in [0, 1]. prefix_frac = 1 is the WORST
    CASE (the whole small answer is produced before escalating, as in a cascade) and is the upper bound on the cost.
    ratio = ((1 - f) c_s + f (prefix_frac c_s + c_l)) / c_l"""
    return ((1 - frac) * c_small + frac * (prefix_frac * c_small + c_large)) / c_large


def breakeven_price_ratio(frac, prefix_frac=1.0):
    """Smallest c_l/c_s at which the router costs no more than always-large: 1 + f k / (1 - f)."""
    return 1 + frac * prefix_frac / (1 - frac) if frac < 1 else math.inf


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


def calibrate_file(path, delta=0.02, seed=0, c_small=1.0, c_large=15.0, prefix_frac=0.2, quiet=False):
    """Real-data entry point. Input: JSON lines {"entropy": float, "small_correct": bool, "large_correct": bool}, one per
    query, produced by running BOTH models on a sample of your traffic (entropy = the feature of 39.1 for the small model).
    Splits 50/50 (seeded), picks the conservative threshold on the validation half, reports on the test half."""
    import json
    rows = [json.loads(l) for l in open(path) if l.strip()]
    data = dict(H=np.array([r["entropy"] for r in rows], float), cs=np.array([r["small_correct"] for r in rows], bool),
                cl=np.array([r["large_correct"] for r in rows], bool))
    val, test = split(data, 0.5, np.random.default_rng(seed))
    tau = calibrate_tau(val, val["H"], delta, True)
    e = evaluate(test, test["H"], tau)
    g = e["gap"]
    se = g.std(ddof=1) / math.sqrt(len(g))
    sigma = float(np.std(evaluate(val, val["H"], tau)["gap"], ddof=1))
    n_needed = int(math.ceil((2 * Z95_ONE_SIDED * sigma / delta) ** 2)) if sigma > 0 else 0
    out = dict(n=len(rows), tau=tau, frac=e["frac"], test_quality=e["quality"], test_gap=float(g.mean()), test_gap_lower95=float(g.mean() - Z95_ONE_SIDED * se),
               always_large=float(test["cl"].mean()), always_small=float(test["cs"].mean()),
               cost_ratio_k1=cost_ratio(e["frac"], c_small, c_large, 1.0), cost_ratio_k=cost_ratio(e["frac"], c_small, c_large, prefix_frac),
               breakeven_k1=breakeven_price_ratio(e["frac"], 1.0), breakeven_k=breakeven_price_ratio(e["frac"], prefix_frac),
               auc_small_wrong=auc(data["H"], ~data["cs"]), validation_n_for_half_delta_margin=n_needed)
    if not quiet:
        for k, v in out.items():
            print(f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v}")
    return out


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
    hand = (0.7 * 1.0 + 0.3 * (1.0 + 15.0)) / 15.0
    check(f"cost ratio by hand, worst case k=1: (0.7*1 + 0.3*(1+15))/15 = {hand:.4f}; function gives {cr:.4f}", abs(cr - hand) < 1e-12)
    cr2 = cost_ratio(0.3, 1.0, 15.0, prefix_frac=0.2)
    hand2 = (0.7 * 1.0 + 0.3 * (0.2 + 15.0)) / 15.0
    check(f"cost ratio by hand, K-token prefix = 20% of a full small answer: (0.7 + 0.3*15.2)/15 = {hand2:.4f}; function gives {cr2:.4f}", abs(cr2 - hand2) < 1e-12)
    check("routing everything costs (k c_s + c_l)/c_l: 1.0667 at k=1 and 1.0133 at k=0.2, never below 1",
          abs(cost_ratio(1.0, 1.0, 15.0) - 16 / 15) < 1e-12 and abs(cost_ratio(1.0, 1.0, 15.0, 0.2) - 15.2 / 15) < 1e-12)
    be1, be2 = breakeven_price_ratio(0.84, 1.0), breakeven_price_ratio(0.84, 0.2)
    check(f"break-even price ratio 1 + f k/(1-f) at f=0.84: {be1:.2f} for k=1, {be2:.2f} for k=0.2; the router costs exactly 1.0 of always-large there",
          abs(be1 - 6.25) < 1e-9 and abs(be2 - 2.05) < 1e-9 and abs(cost_ratio(0.84, 1.0, be2, 0.2) - 1.0) < 1e-9)

    rng = np.random.default_rng(1)
    val, test = split(make_data(30000, seed=2), 0.5, rng)
    tau = calibrate_tau(val, val["H"], delta=0.02, conservative=True)
    gv = evaluate(val, val["H"], tau)["gap"].mean()
    check(f"(weak, by construction) the chosen threshold meets its own criterion on validation: gap {gv:+.4f} >= -0.02 (tau={tau:.3f})", gv >= -0.02 - 1e-9)
    check("(a theorem, since SE >= 0) the conservative threshold is no higher than the naive one",
          calibrate_tau(val, val["H"], conservative=True) <= calibrate_tau(val, val["H"], conservative=False) + 1e-12)
    # the real test: does the chosen tau generalise? score it on a large FRESH set
    fresh = make_data(100000, seed=555)
    vc = vn = 0
    NS = 100
    for k in range(NS):
        d = make_data(4000, seed=7000 + k)
        v, _ = split(d, 0.5, np.random.default_rng(k))
        gc = evaluate(fresh, fresh["H"], calibrate_tau(v, v["H"], 0.02, True))["gap"].mean()
        gn = evaluate(fresh, fresh["H"], calibrate_tau(v, v["H"], 0.02, False))["gap"].mean()
        vc += gc < -0.02; vn += gn < -0.02
    check(f"GENERALISATION over {NS} splits, scored on 100k fresh queries: conservative violates the 2-point target in {vc}/{NS} ({vc / NS:.2f}), naive in {vn}/{NS} ({vn / NS:.2f}); asserted: conservative <= 0.12 and strictly fewer than naive",
          vc / NS <= 0.12 and vc < vn)
    # calibrate CLI on an exported file
    import json, tempfile, os
    ex = make_data(3000, seed=9)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        for i in range(3000):
            fh.write(json.dumps(dict(entropy=float(ex["H"][i]), small_correct=bool(ex["cs"][i]), large_correct=bool(ex["cl"][i]))) + "\n")
        path = fh.name
    out = calibrate_file(path, delta=0.02, seed=0, c_small=1.0, c_large=15.0, prefix_frac=0.2, quiet=True)
    os.unlink(path)
    check(f"calibrate_file() on an exported jsonl: n={out['n']}, tau={out['tau']:.3f}, routed {out['frac']:.2f}, test gap {out['test_gap']:+.4f}", out["n"] == 3000 and out["tau"] > -math.inf and 0 < out["frac"] <= 1)
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

    print("\n== 3. Calibration: naive vs conservative. Each chosen tau is scored on (a) its own 2000-query test half and (b) a 100 000-query FRESH set ==")
    fresh = make_data(100000, seed=999)
    print(f"  {'validation n':>12s} {'splits':>6s} {'method':13s} {'P(test-half gap < -.02)':>24s} {'P(fresh-set gap < -.02)':>24s} {'mean fresh gap':>15s} {'routed':>7s}")
    for nval, splits_n in ((1000, 300), (2000, 300), (5000, 120), (20000, 40)):
        out = {"naive": [], "conservative": []}
        for k in range(splits_n):
            d = make_data(2 * nval, seed=1000 + k)
            v, t = split(d, 0.5, np.random.default_rng(k))
            for mode in out:
                tau = calibrate_tau(v, v["H"], 0.02, conservative=(mode == "conservative"))
                out[mode].append((evaluate(t, t["H"], tau)["gap"].mean(), evaluate(fresh, fresh["H"], tau)["gap"].mean(), evaluate(fresh, fresh["H"], tau)["frac"]))
        for mode, arr in out.items():
            a_ = np.array(arr)
            print(f"  {nval:12d} {splits_n:6d} {mode:13s} {np.mean(a_[:, 0] < -0.02):24.3f} {np.mean(a_[:, 1] < -0.02):24.3f} {a_[:, 1].mean():15.4f} {a_[:, 2].mean():7.3f}")
    g = evaluate(D, D["H"], 1.4)["gap"]
    print(f"  per-query paired gap std (tau=1.4): {g.std(ddof=1):.3f}  -> to make 1.645*SE <= delta/2 = 0.01 you need about {int(math.ceil((2 * Z95_ONE_SIDED * g.std(ddof=1) / 0.02) ** 2))} validation queries")

    print("\n== 4. Distribution shift: calibrate on Beta(2,3) difficulty, deploy on harder traffic ==")
    v = make_data(30000, seed=7)
    tau = calibrate_tau(v, v["H"], 0.02, True)
    print(f"  calibrated tau = {tau:.3f} nats")
    for name, (a, b) in (("same traffic Beta(2,3)", (2, 3)), ("harder Beta(3,2)", (3, 2)), ("much harder Beta(5,2)", (5, 2)), ("easier Beta(2,5)", (2, 5))):
        t = make_data(30000, seed=8, a=a, b=b)
        e = evaluate(t, t["H"], tau)
        un = t["H"] <= tau
        print(f"  {name:24s}: routed {e['frac']:.3f}  quality {e['quality']:.3f}  always-large {t['cl'].mean():.3f}  gap {e['gap'].mean():+.4f}   unrouted slice {un.mean():.3f}: small-model error rate in it {1 - t['cs'][un].mean():.3f}, large-model {1 - t['cl'][un].mean():.3f}")

    print("\n== 5. When does the router pay? cost ratio at the calibrated routed fraction, worst case (k=1: whole small answer wasted) and K-token prefix = 20% of an answer (k=0.2) ==")
    tau = calibrate_tau(v, v["H"], 0.02, True)
    f = evaluate(D, D["H"], tau)["frac"]
    print(f"  calibrated routed fraction on the 60k set: {f:.3f}; break-even price ratio c_l/c_s = 1 + f k/(1-f): {breakeven_price_ratio(f, 1.0):.2f} (k=1), {breakeven_price_ratio(f, 0.2):.2f} (k=0.2)")
    print(f"  {'c_l/c_s':>8s} {'cost k=1':>9s} {'cost k=0.2':>11s}")
    for r in (2, 5, 10, 15, 30, 100):
        print(f"  {r:8d} {cost_ratio(f, 1.0, float(r), 1.0):9.3f} {cost_ratio(f, 1.0, float(r), 0.2):11.3f}")
    print("  (the router also adds latency to every ROUTED query: the small model's prefill plus K decode steps before escalation)")
    print("\n== 6. What does an entropy of ~1.4 nats look like in this synthetic world? mean over 4000 draws of the per-position distribution ==")
    rng = np.random.default_rng(3)
    for boost in (7.0, 6.0, 5.0, 4.0, 3.0):
        lg = rng.standard_normal((4000, V)); lg[:, 0] += boost
        Hs = token_entropy_nats(lg); z = lg - lg.max(-1, keepdims=True); p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
        top = p.max(-1); srt = -np.sort(-p, -1)
        print(f"  boost {boost:3.1f}: H = {Hs.mean():.2f} nats, top-token probability {top.mean():.2f}, mass beyond the top 5 tokens {1 - srt[:, :5].sum(-1).mean():.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("path")
        ap.add_argument("--delta", type=float, default=0.02); ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--c-small", type=float, default=1.0); ap.add_argument("--c-large", type=float, default=15.0)
        ap.add_argument("--prefix-frac", type=float, default=0.2)
        a_ = ap.parse_args()
        calibrate_file(a_.path, a_.delta, a_.seed, a_.c_small, a_.c_large, a_.prefix_frac)
        sys.exit(0)
    t0 = time.time()
    print("self-tests:")
    ok = selftests()
    if "--selftest" not in sys.argv:
        experiments()
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}   ({time.time() - t0:.0f} s)")
    sys.exit(0 if ok else 1)
