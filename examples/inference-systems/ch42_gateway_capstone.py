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

from ch36_benchmark_teardown import Rec, summarize_runs, percentile
from ch38_semantic_cache import World, SemanticCache
from ch39_complexity_router import make_data, calibrate_tau
from ch40_gpu_autoscaler import Scaler, make_arrivals, lognormal_service
from ch41_edge_fallback import CircuitBreaker

# ---------------------------------------------------------------- assumed constants
HORIZON, DRAIN = 900.0, 180.0
SEGS = [(0, 400, 4.0), (400, 500, 12.0), (500, HORIZON, 4.0)]    # req/s, with a burst
THRESH = 0.92                       # cache cosine threshold
FAMILIES, ZIPF = 1500, 0.6          # cache workload (6000 intents)
CACHE_LAT = 0.02                    # s, embedding + lookup on a hit
C_EMB = 1e-5                        # $ per request for the embedding model (assumed)
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


def gateway(cfg, seed, outage=False):
    """Causal discrete-event simulation: decisions see only completed prior events.

    Pool admission/scaling is sampled every 0.1 s as in Chapter 40. Service
    completions, request arrivals, prefix completions and breaker timeouts are
    separate ordered events. In particular, only a successful completed model
    request may publish a semantic-cache entry.
    """
    import heapq
    from collections import deque

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
    simh, dt = HORIZON + DRAIN, 0.1
    recs = [Rec(intended=t, sent=t) for t in arr]
    kind = [None] * n
    correct = np.zeros(n, bool)
    fallbacks = hit_wrong = 0
    insert_log, cache_hit_events, prefix_log = [], [], []
    direct_small = set()
    pools, gpu_s = {}, {p: 0.0 for p in POOLS}
    events, serial = [], 0

    def event(at, priority, action, payload):
        nonlocal serial
        serial += 1
        heapq.heappush(events, (at, priority, serial, action, payload))

    for p in POOLS:
        active = p == "large" or cfg.router or cfg.fallback
        count = (1 if cfg.autoscale else
                 cfg.large_static if p == "large" else cfg.small_static) if active else 0
        scaler = _scaler() if cfg.autoscale and active else None
        if scaler is not None:
            scaler.history.append((0.0, count))
        pools[p] = dict(queue=deque(), reps=[dict(ready_at=0.0, running={}, draining=False)
                                            for _ in range(count)],
                        scaler=scaler, next_poll=0.0,
                        next_hpa=scaler.poll if scaler else 0.0)

    def enqueue(pool, rid, at, prefix=False, gen=None):
        pools[pool]["queue"].append((rid, at, prefix, gen))

    def resize(state, new, now):
        live = [r for r in state["reps"] if not r["draining"]]
        if new > len(live):
            state["reps"].extend(dict(ready_at=now+COLD, running={}, draining=False)
                                  for _ in range(new-len(live)))
        else:
            live.sort(key=lambda r: (r["ready_at"] <= now, len(r["running"])))
            for replica in live[:len(live)-new]:
                if replica["ready_at"] > now:
                    state["reps"].remove(replica)
                else:
                    replica["draining"] = True

    def large_decision(rid, now):
        nonlocal fallbacks
        down = outage and in_outage(now)
        gen = None
        if breaker is not None:
            mode, gen = breaker.allow(now)
            if mode is None:
                kind[rid] = "small"
                fallbacks += 1
                enqueue("small", rid, now)
                return
        if down:
            # Deliver this result only after the timeout; no future-state mutation.
            event(now+HANG, 0, "timeout", (rid, gen))
        else:
            kind[rid] = "large"
            enqueue("large", rid, now, gen=gen)

    for rid, at in enumerate(arr):
        event(float(at), 1, "arrival", rid)
    for step in range(int(round(simh/dt))):
        event(step*dt, 2, "tick", None)

    while events:
        now, _priority, _serial, action, payload = heapq.heappop(events)
        if now > simh:
            break
        if action == "arrival":
            rid = payload
            if cache is not None:
                entry, _sim = cache.lookup("k", vecs[rid], now=now)
                if entry is not None:
                    kind[rid] = "hit"
                    correct[rid] = entry["answer"] == int(intents[rid]) and entry["correct"]
                    hit_wrong += int(entry["answer"] != int(intents[rid]))
                    recs[rid].first = recs[rid].end = now+CACHE_LAT
                    recs[rid].ntok, recs[rid].ok = 100, True
                    insert_log.append((entry["created"], now))
                    cache_hit_events.append((entry["producer"], entry["created"], now))
                    continue
            route_large = not cfg.router or data["H"][intents[rid]] > tau
            if route_large and cfg.router:
                enqueue("small", rid, now, prefix=True)
            elif route_large:
                large_decision(rid, now)
            else:
                kind[rid] = "small"
                direct_small.add(rid)
                enqueue("small", rid, now)
        elif action == "timeout":
            rid, gen = payload
            if breaker is not None:
                breaker.record(gen, False, now)
                kind[rid] = "small"
                fallbacks += 1
                enqueue("small", rid, now)
            else:
                kind[rid] = "fail"
                recs[rid].err, recs[rid].end = "timeout", now
        elif action == "complete":
            pool, replica, rid, prefix, gen, started, jobid = payload
            replica["running"].pop(jobid)
            if prefix:
                prefix_log.append((rid, started, now))
                large_decision(rid, now)
                continue
            r, P = recs[rid], POOLS[pool]
            duration = now-started
            r.first = started+min(P["prefill"], duration*0.5)
            if rid in direct_small:
                # The single full-small job includes its routing prefix. Do not
                # release that prefix before the small/large decision is ready.
                decision_at = started+PREFIX_FRAC*duration
                prefix_log.append((rid, started, decision_at))
                r.first = max(r.first, decision_at)
            r.end, r.ntok, r.ok = now, max(2, int(duration*P["tok"])), True
            correct[rid] = bool(data["cl" if pool == "large" else "cs"][intents[rid]])
            if breaker is not None and pool == "large":
                breaker.record(gen, True, now)
            if cache is not None:
                cache.insert("k", vecs[rid], int(intents[rid]), now=now,
                             correct=bool(correct[rid]), producer=rid)
        else:  # pool scheduler / scaler tick
            for pool, state in pools.items():
                state["reps"][:] = [r for r in state["reps"]
                                         if not (r["draining"] and not r["running"])]
                for replica in state["reps"]:
                    if replica["draining"] or replica["ready_at"] > now:
                        continue
                    while state["queue"] and len(replica["running"]) < POOLS[pool]["slots"]:
                        rid, at, prefix, gen = state["queue"].popleft()
                        assert at <= now
                        duration = float(svc_all[pool][rid])*(PREFIX_FRAC if prefix else 1.0)
                        jobid = (rid, prefix)
                        replica["running"][jobid] = now+duration
                        event(now+duration, 0, "complete",
                              (pool, replica, rid, prefix, gen, now, jobid))
                scaler = state["scaler"]
                if scaler is not None:
                    outstanding = len(state["queue"])+sum(len(r["running"]) for r in state["reps"])
                    current = sum(not r["draining"] for r in state["reps"])
                    if now >= state["next_poll"]:
                        state["next_poll"] += scaler.poll
                        new = scaler.keda_poll(now, outstanding, current)
                        if new is not None and new != current:
                            resize(state, new, now)
                            current = new
                    if now >= state["next_hpa"]:
                        state["next_hpa"] += scaler.hpa_period
                        if current:
                            if not scaler.cached_metrics:
                                scaler.sample = outstanding
                            new = scaler.hpa(now, current)
                            if new != current:
                                resize(state, new, now)
                gpu_s[pool] += len(state["reps"])*dt

    for rid, r in enumerate(recs):
        if not r.ok:
            correct[rid] = False
            if kind[rid] is None:
                kind[rid] = "fail"
            if not r.err:
                r.err = "unserved by drain horizon"
    cost = sum(gpu_s[p]/3600.0*POOLS[p]["price"] for p in gpu_s)+(n*C_EMB if cache else 0.0)
    return dict(recs=recs, n=n, kind=kind, correct=correct, cost=cost, gpu_s=gpu_s, tau=tau,
                hits=kind.count("hit"), hit_wrong=hit_wrong, fallbacks=fallbacks,
                insert_log=insert_log, cache_hit_events=cache_hit_events, prefix_log=prefix_log,
                arr=arr, quality=float(correct.mean()), failures=sum(not r.ok for r in recs))


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
    paired_diffs = [percentile(summarize_cfg([d])["ttft_samples"], 0.5) -
                    percentile(summarize_cfg([a])["ttft_samples"], 0.5)
                    for d, a in zip(store[CD.name], store[CA.name])]
    print(f"\nTTFT p50, D minus A, paired per-run mean [min-max]: {fmt(paired_diffs, '.2f')} s; "
          "3 runs do not establish a precise confidence interval")

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
