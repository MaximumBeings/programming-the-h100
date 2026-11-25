#!/usr/bin/env python3
"""Chapter 42 -- Capstone: one gateway, four mechanisms, one measurement harness (CPU only).

Request path:   semantic cache (ch38) -> complexity router (ch39) -> small / large GPU pools
                with queue-based autoscaling (ch40) -> circuit breaker + fallback (ch41).
Measurement:    Rec records and summarize_runs / diff_ci / percentile from ch36.

Everything below the module imports is an ASSUMED model; nothing was run on a GPU.

    python3 ch42_gateway_capstone.py            # tests + experiments
    python3 ch42_gateway_capstone.py tests
"""
import sys
from dataclasses import dataclass, replace

import numpy as np

from ch36_benchmark_teardown import Rec, summarize_runs, diff_ci, percentile
from ch38_semantic_cache import World, SemanticCache
from ch39_complexity_router import make_data, calibrate_tau
from ch40_gpu_autoscaler import Scaler, simulate, make_arrivals, lognormal_service
from ch41_edge_fallback import CircuitBreaker

# ---------------------------------------------------------------- assumed constants
HORIZON, DRAIN = 900.0, 180.0
SEGS = [(0, 400, 4.0), (400, 500, 12.0), (500, HORIZON, 4.0)]    # req/s, with a burst
THRESH = 0.92                       # cache cosine threshold
FAMILIES, ZIPF = 1500, 0.6          # cache workload (6000 intents)
CACHE_LAT = 0.02                    # s, embedding + lookup on a hit
C_EMB = 1e-5                        # $ per request for the embedding model (assumed)
ROUTER_DELAY = 0.15                 # s added to a request routed to the large pool
PREFIX_FRAC = 0.2                   # the router's K-token prefix costs this fraction of a small answer
OUTAGE = (600.0, 720.0)             # the large pool is unreachable
HANG = 3.0                          # client waits this long for a hung large pool
COLD = 60.0
SLO_TTFT, SLO_TPOT = 5.0, 0.05
POOLS = {
    "small": dict(mean=0.6, sigma=0.5, slots=4, price=1.0, tok=40.0, prefill=0.15),
    "large": dict(mean=3.0, sigma=0.5, slots=4, price=4.0, tok=30.0, prefill=0.40),
}


@dataclass(frozen=True)
class Cfg:
    name: str
    cache: bool = False
    router: bool = False
    autoscale: bool = False
    fallback: bool = False
    large_static: int = 12
    small_static: int = 2
    thresh: float = THRESH


def in_outage(t):
    return OUTAGE[0] <= t < OUTAGE[1]


def _scaler():
    return Scaler(target=3.0, min_replicas=1, max_replicas=40)


def run_pool(name, jobs, svc_all, cfg, simh):
    """jobs: list of (pool_arrival, key). Returns (per-key result dict, gpu_s)."""
    P = POOLS[name]
    jobs = sorted(jobs)
    arrs = np.array([a for a, _ in jobs])
    svc = np.array([svc_all[name][k] for _, k in jobs])
    static = cfg.large_static if name == "large" else cfg.small_static
    kw = dict(scaler=_scaler()) if cfg.autoscale else dict(static=static)
    res = simulate(arrs, svc, simh, slots=P["slots"], cold=COLD, **kw)
    out = {}
    for pos, (a, k) in enumerate(jobs):
        if res["started"][pos]:
            out[k] = (res["started_at"][pos], res["finished"][pos])
    return out, res["gpu_s"], res


def gateway(cfg, seed, outage=False):
    rng = np.random.default_rng(seed)
    arr = make_arrivals(SEGS, rng)
    n = len(arr)
    world = World(seed=seed, families=FAMILIES)
    intents, vecs = world.stream(n, zipf=ZIPF)
    data = make_data(world.n_intents, seed + 100)
    val = make_data(2000, seed + 200)
    tau = calibrate_tau(val, val["H"], delta=0.02)
    if cfg.router == "all_large":
        tau = -np.inf
    svc_all = {p: lognormal_service(n, POOLS[p]["mean"], POOLS[p]["sigma"],
                                    np.random.default_rng(seed * 31 + j))
               for j, p in enumerate(("small", "large"))}
    cache = SemanticCache(world.dim, cfg.thresh) if cfg.cache else None
    breaker = CircuitBreaker(3, 10.0) if cfg.fallback else None
    pending, seq = [], 0
    kind = [None] * n          # hit | large | small | fail
    correct = np.zeros(n, bool)
    jobs = {"small": [], "large": []}
    t_in = np.array(arr, float)    # time the request enters its pool
    fail_end = {}
    fallbacks = hit_wrong = 0
    insert_log = []                # (insert_time, hit_time) pairs for the ordering invariant
    import heapq
    for i in range(n):
        t = arr[i]
        while pending and pending[0][0] <= t:
            ts, _s, key, vec, ans, ok = heapq.heappop(pending)
            cache.insert(key, vec, ans, now=ts, correct=ok)
        intent = int(intents[i])
        if cache is not None:
            entry, _sim = cache.lookup("k", vecs[i], now=t)
            if entry is not None:
                kind[i] = "hit"
                correct[i] = (entry["answer"] == intent) and entry["correct"]
                hit_wrong += int(entry["answer"] != intent)
                insert_log.append((entry["created"], t))
                continue
        pool = "large" if (not cfg.router or data["H"][intent] > tau) else "small"
        delay = 0.0
        if pool == "large" and cfg.router:
            delay += ROUTER_DELAY
            jobs["small"].append((t, ("prefix", i)))       # the router's K-token prefix runs on the small pool
        if pool == "large":
            down = outage and in_outage(t)
            if breaker is not None:
                mode, gen = breaker.allow(t)
                if mode is None:
                    pool, fallbacks = "small", fallbacks + 1
                else:
                    breaker.record(gen, not down, t + (HANG if down else 0.0))
                    if down:
                        pool, delay, fallbacks = "small", delay + HANG, fallbacks + 1
            elif down:
                kind[i] = "fail"
                fail_end[i] = t + HANG
                continue
        kind[i] = pool
        t_in[i] = t + delay
        jobs[pool].append((t + delay, ("req", i)))
        correct[i] = bool(data["cl" if pool == "large" else "cs"][intent])
        if cache is not None:
            seq += 1
            heapq.heappush(pending, (t + delay + POOLS[pool]["mean"], seq, "k", vecs[i], intent,
                                     bool(correct[i])))
    # pools
    simh = HORIZON + DRAIN
    key_svc = {p: _KeySvc(svc_all[p], POOLS["small"]["mean"] * PREFIX_FRAC / POOLS["small"]["mean"])
               for p in svc_all}
    results, gpu_s = {}, {}
    for p in ("small", "large"):
        exists = bool(jobs[p]) or cfg.autoscale and (p == "large" or cfg.router or cfg.fallback)
        if p == "small" and not (cfg.router or cfg.fallback):
            exists = False
        if exists:
            results[p], gpu_s[p], _ = run_pool(p, jobs[p], key_svc, cfg, simh)
        else:
            results[p], gpu_s[p] = {}, 0.0
    recs = []
    for i in range(n):
        r = Rec(intended=arr[i], sent=arr[i])
        if kind[i] == "hit":
            r.first = r.end = arr[i] + CACHE_LAT
            r.ntok, r.ok = 100, True
        elif kind[i] == "fail":
            r.err, r.end = "timeout", fail_end[i]
        else:
            res = results[kind[i]].get(("req", i))
            if res is None:
                r.err = "unserved"
            else:
                P = POOLS[kind[i]]
                st, fin = res
                svc = fin - st
                r.first = st + min(P["prefill"], svc * 0.5)
                r.end = fin
                r.ntok = max(2, int(svc * P["tok"]))
                r.ok = True
        recs.append(r)
        if not r.ok:
            correct[i] = False
    cost = sum(gpu_s[p] / 3600.0 * POOLS[p]["price"] for p in gpu_s) + (n * C_EMB if cache else 0.0)
    return dict(recs=recs, n=n, kind=kind, correct=correct, cost=cost, gpu_s=gpu_s, tau=tau,
                hits=sum(1 for k in kind if k == "hit"), hit_wrong=hit_wrong,
                fallbacks=fallbacks, insert_log=insert_log, arr=arr,
                quality=float(correct.mean()),
                failures=sum(1 for r in recs if not r.ok))


class _KeySvc(dict):
    """Service time lookup by ('req', i) or ('prefix', i); the prefix is a fraction of a small answer."""
    def __init__(self, arr, frac):
        super().__init__()
        self.arr, self.frac = arr, frac

    def __getitem__(self, k):
        kind, i = k
        return self.arr[i] * (PREFIX_FRAC if kind == "prefix" else 1.0)


def summarize_cfg(runs):
    s = summarize_runs([r["recs"] for r in runs], warmup=10, slo_ttft=SLO_TTFT, slo_tpot=SLO_TPOT)
    return s


# ---------------------------------------------------------------- tests
CA = Cfg("A all-large, static 12")
CB = Cfg("B +cache, static 12", cache=True)
CB2 = Cfg("B2 +cache, static 7", cache=True, large_static=7)
CC = Cfg("C +router, static 12+2", cache=True, router=True)
CC2 = Cfg("C2 +router, static 6+1", cache=True, router=True, large_static=6, small_static=1)
CF = Cfg("F autoscaler only", autoscale=True)
CD = Cfg("D cache+router+autoscale", cache=True, router=True, autoscale=True)
LADDER = [CA, CB, CB2, CC, CC2, CF, CD]


def selftests():
    out = []

    def t(name):
        def deco(f):
            f()
            out.append(name)
        return deco

    base = gateway(CA, 1)

    @t("1 every request has exactly one fate; counts add up")
    def _():
        g = gateway(CC, 1)
        assert len(g["kind"]) == g["n"] and all(k in ("hit", "large", "small", "fail") for k in g["kind"])
        assert sum(1 for k in g["kind"] if k in ("hit", "large", "small", "fail")) == g["n"]

    @t("2 all-large, no cache: quality equals the mean large-model correctness of the drawn intents")
    def _():
        g = base
        rng = np.random.default_rng(1)
        arr = make_arrivals(SEGS, rng)
        world = World(seed=1, families=FAMILIES)
        intents, _v = world.stream(len(arr), zipf=ZIPF)
        data = make_data(world.n_intents, 101)
        assert g["failures"] == 0
        assert abs(g["quality"] - float(np.mean(data["cl"][intents]))) < 1e-12

    @t("3 router: tau=-inf routes everything large; cache off gives the all-large quality")
    def _():
        g = gateway(Cfg("x", router="all_large"), 1)
        assert g["kind"].count("small") == 0 and g["kind"].count("large") == g["n"]
        assert abs(g["quality"] - base["quality"]) < 1e-12

    @t("4 a threshold of 1.0 gives only exact-repeat hits, none of them wrong")
    def _():
        g = gateway(Cfg("x", cache=True, thresh=1.0 - 1e-12), 1)
        assert g["hit_wrong"] == 0 and g["hits"] > 0

    @t("5 no hit is served before its entry was inserted")
    def _():
        g = gateway(CB, 2)
        assert g["insert_log"] and all(ins <= hit for ins, hit in g["insert_log"])

    @t("6 deterministic for a fixed seed")
    def _():
        a, b = gateway(CD, 3), gateway(CD, 3)
        assert a["cost"] == b["cost"] and a["quality"] == b["quality"] and a["kind"] == b["kind"]

    @t("7 a request is never served before it arrived; first token before end")
    def _():
        g = gateway(CD, 2)
        for r in g["recs"]:
            if r.ok:
                assert r.first >= r.intended and r.end >= r.first

    @t("8 static all-large cost equals replicas x simulated time x price")
    def _():
        expect = 12 * (HORIZON + DRAIN) / 3600.0 * POOLS["large"]["price"]
        assert abs(base["cost"] - expect) / expect < 0.01

    @t("9 no outage: the fallback policy changes nothing")
    def _():
        a = gateway(Cfg("x", cache=True, router=True, fallback=True), 1)
        b = gateway(Cfg("x", cache=True, router=True), 1)
        assert a["kind"] == b["kind"] and a["fallbacks"] == 0

    @t("10 outage: fallback strictly reduces failures; without it every outage-time large request fails")
    def _():
        c = Cfg("x", cache=True, router=True, large_static=12)
        no = gateway(c, 1, outage=True)
        fb = gateway(replace(c, fallback=True), 1, outage=True)
        assert no["failures"] > fb["failures"]
        assert no["failures"] == sum(1 for k, a in zip(no["kind"], no["arr"]) if k == "fail")

    @t("11 SLO attainment never exceeds success rate (failures stay in the denominator)")
    def _():
        s = summarize_cfg([gateway(CC, 1, outage=True)])
        assert s["slo_attain"] <= s["success_rate"] + 1e-12

    @t("12 cache hits lower the load: fewer requests reach the pools")
    def _():
        g = gateway(CB, 1)
        assert g["hits"] > 0.3 * g["n"]

    return out


# ---------------------------------------------------------------- experiments
def fmt(v, spec=".1f", scale=1.0):
    v = [x * scale for x in v]
    return f"{sum(v) / len(v):{spec}} [{min(v):{spec}}-{max(v):{spec}}]"


def run_cfg(cfg, seeds, outage=False):
    runs = [gateway(cfg, s, outage) for s in seeds]
    per = [summarize_cfg([r]) for r in runs]
    return runs, per


def experiments():
    seeds = (0, 1, 2)
    print("\n=== Experiment 1: the ladder, no outage (mean [min-max], 3 seeds) ===")
    print(f"{'config':26s} {'$/1k req':>16s} {'SLO attain %':>16s} {'ttft p50 s':>14s} {'ttft p99 s':>16s} "
          f"{'quality %':>16s} {'hit %':>8s} {'large %':>8s}")
    store = {}
    for c in LADDER:
        runs, per = run_cfg(c, seeds)
        store[c.name] = runs
        n = [r["n"] for r in runs]
        print(f"{c.name:26s} "
              f"{fmt([r['cost'] / r['n'] * 1000 for r in runs], '.3f'):>16s} "
              f"{fmt([p['slo_attain'] for p in per], '.1f', 100):>16s} "
              f"{fmt([p['ttft_p50'] for p in per], '.2f'):>14s} "
              f"{fmt([p['ttft_p99'] for p in per], '.2f'):>16s} "
              f"{fmt([r['quality'] for r in runs], '.1f', 100):>16s} "
              f"{100 * sum(r['hits'] for r in runs) / sum(n):8.1f} "
              f"{100 * sum(r['kind'].count('large') for r in runs) / sum(n):8.1f}")
    sa = summarize_cfg(store[CA.name]); sd = summarize_cfg(store[CD.name])
    lo, hi = diff_ci(sd["ttft_samples"], sa["ttft_samples"], lambda x: percentile(x, 0.5))
    print(f"\nTTFT p50, D minus A, pooled: 95% bootstrap CI [{lo:.2f}, {hi:.2f}] s")

    print("\n=== Experiment 2: large pool unreachable for 120 s (t = 600-720), config D ===")
    D = CD
    print(f"{'policy':26s} {'failures':>14s} {'SLO attain %':>16s} {'quality %':>16s} {'$/1k req':>16s} {'fallbacks':>12s}")
    for label, c in (("no fallback", D), ("breaker + small-pool fallback", replace(D, fallback=True))):
        runs, per = run_cfg(c, seeds, outage=True)
        print(f"{label:26s} {fmt([r['failures'] for r in runs], '.0f'):>14s} "
              f"{fmt([p['slo_attain'] for p in per], '.1f', 100):>16s} "
              f"{fmt([r['quality'] for r in runs], '.1f', 100):>16s} "
              f"{fmt([r['cost'] / r['n'] * 1000 for r in runs], '.3f'):>16s} "
              f"{fmt([r['fallbacks'] for r in runs], '.0f'):>12s}")

    print("\n=== Experiment 3: cache threshold, end to end (config D) ===")
    print(f"{'threshold':>10s} {'hit %':>14s} {'wrong-hit % of hits':>22s} {'quality %':>16s} {'$/1k req':>16s}")
    for th in (0.85, 0.92, 0.97):
        runs, _ = run_cfg(replace(D, thresh=th), seeds)
        print(f"{th:10.2f} {fmt([100 * r['hits'] / r['n'] for r in runs], '.1f'):>14s} "
              f"{fmt([100 * r['hit_wrong'] / max(1, r['hits']) for r in runs], '.1f'):>22s} "
              f"{fmt([r['quality'] for r in runs], '.1f', 100):>16s} "
              f"{fmt([r['cost'] / r['n'] * 1000 for r in runs], '.3f'):>16s}")
    runs, _ = run_cfg(replace(D, cache=False), seeds)
    print(f"{'no cache':>10s} {'0.0':>14s} {'-':>22s} {fmt([r['quality'] for r in runs], '.1f', 100):>16s} "
          f"{fmt([r['cost'] / r['n'] * 1000 for r in runs], '.3f'):>16s}")


if __name__ == "__main__":
    names = selftests()
    for nme in names:
        print("PASS", nme)
    print(f"{len(names)} tests passed")
    if "tests" not in sys.argv[1:]:
        experiments()
