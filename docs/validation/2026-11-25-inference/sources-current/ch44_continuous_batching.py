#!/usr/bin/env python3
"""Chapter 44 - A continuous-batching scheduler with preemption (CPU simulator).

Run:  python3 ch44_continuous_batching.py          # self-tests + experiments
      python3 ch44_continuous_batching.py tests    # self-tests only

Everything is simulated. The per-iteration cost model and the workload are ASSUMED.
"""
import math
import sys
import time
from collections import deque

import numpy as np

# ---- assumed cost model: seconds per engine iteration ----
T0 = 0.015       # fixed cost (weights streamed once per iteration)
CP = 0.00005     # per prefill token
CD = 0.0002      # per decoding sequence


class Req:
    __slots__ = ("id", "arrival", "prompt", "out", "generated", "computed", "target", "blocks",
                 "first_token", "done", "preempts", "rejected")

    def __init__(self, id, arrival, prompt, out):
        self.id, self.arrival, self.prompt, self.out = id, arrival, prompt, out
        self.generated = 0
        self.computed = 0
        self.target = prompt
        self.blocks = 0
        self.first_token = None
        self.done = None
        self.preempts = 0
        self.rejected = False


def make_workload(seed, n, rate, pmed=256, psig=0.8, omed=128, osig=1.0, pmax=2048, omax=1024):
    r = np.random.default_rng(seed)
    arr = np.cumsum(r.exponential(1.0 / rate, n))
    prompts = np.clip(np.rint(np.exp(r.normal(math.log(pmed), psig, n))), 16, pmax).astype(int)
    outs = np.clip(np.rint(np.exp(r.normal(math.log(omed), osig, n))), 1, omax).astype(int)
    return [Req(i, float(arr[i]), int(prompts[i]), int(outs[i])) for i in range(n)]


def clone(reqs):
    return [Req(q.id, q.arrival, q.prompt, q.out) for q in reqs]


# --------------------------------------------------------------------------
# Continuous batching
# --------------------------------------------------------------------------
def simulate_continuous(reqs, n_blocks=2048, B=16, max_seqs=128, max_batched_tokens=None,
                        watermark=0.01, debug=False, t0=T0, cp=CP, cd=CD):
    wm = int(math.ceil(watermark * n_blocks))
    budget_cap = max_batched_tokens if max_batched_tokens else 10 ** 12
    free = n_blocks
    waiting = deque()
    running = []          # admission order, oldest first
    pending = sorted(reqs, key=lambda q: q.arrival)
    ip = 0
    t = 0.0
    finished = rejected = 0
    n = len(pending)
    stats = {"iters": 0, "prefill_tokens": 0, "preemptions": 0, "itl": [], "batch": [], "util": []}

    def need(tokens):
        return -(-tokens // B)

    def release(q):
        nonlocal free
        free += q.blocks
        q.blocks = 0

    def preempt(q):
        running.remove(q)
        release(q)
        q.computed = 0
        q.target = q.prompt + q.generated
        q.preempts += 1
        stats["preemptions"] += 1
        waiting.appendleft(q)

    while finished + rejected < n:
        # arrivals
        if not running and not waiting and ip < n and pending[ip].arrival > t:
            t = pending[ip].arrival
        while ip < n and pending[ip].arrival <= t:
            q = pending[ip]
            ip += 1
            if need(q.prompt + q.out) > n_blocks - wm:   # could never finish: reject up front
                q.rejected = True
                rejected += 1
            else:
                waiting.append(q)
        if not running and not waiting:
            continue

        preempted_now = False
        # 1. decode sequences: each needs a block when its KV length is at a block boundary
        decoders = []
        for q in list(running):
            if q not in running or q.computed < q.target:
                continue
            if q.computed % B == 0:
                while free == 0 and q in running:
                    victim = running[-1]
                    preempt(victim)
                    preempted_now = True
                if q not in running:
                    continue
                free -= 1
                q.blocks += 1
            decoders.append(q)
        # 2. prefill under the token budget
        budget = budget_cap - len(decoders)
        chunks = []
        for q in running:
            if q.computed < q.target and budget > 0:
                c = min(q.target - q.computed, budget)
                chunks.append((q, c))
                budget -= c
        if not preempted_now:
            while waiting and len(running) < max_seqs and budget > 0:
                q = waiting[0]
                nb = need(q.target)
                if nb > free - wm:
                    break                       # strict FCFS: the head blocks the queue
                waiting.popleft()
                free -= nb
                q.blocks = nb
                running.append(q)
                c = min(q.target - q.computed, budget)
                chunks.append((q, c))
                budget -= c
        pf = sum(c for _, c in chunks)
        dt = t0 + cp * pf + cd * len(decoders)
        assert pf + len(decoders) > 0, "scheduler made no progress"
        t += dt
        stats["iters"] += 1
        stats["prefill_tokens"] += pf
        stats["batch"].append(len(running))
        stats["util"].append(1 - free / n_blocks)
        if decoders:
            stats["itl"].append((dt, len(decoders)))
        for q in decoders:
            q.computed += 1
            q.generated += 1
            if q.generated == q.out:
                q.done = t
                running.remove(q)
                release(q)
                finished += 1
        for q, c in chunks:
            q.computed += c
            if q.computed == q.target:
                q.generated += 1
                if q.first_token is None:
                    q.first_token = t
                if q.generated == q.out:
                    q.done = t
                    running.remove(q)
                    release(q)
                    finished += 1
        if debug:
            used = sum(q.blocks for q in running)
            assert used == n_blocks - free and free >= 0
            for q in running:
                assert q.blocks == need(q.computed) or q.computed < q.target and q.blocks == need(q.target), (q.blocks, q.computed, q.target)
    assert free == n_blocks, "blocks leaked"
    stats["end"] = t
    return stats


# --------------------------------------------------------------------------
# Static batching (streaming completion, memory ignored: favourable to static)
# --------------------------------------------------------------------------
def simulate_static(reqs, N, t0=T0, cp=CP, cd=CD):
    pending = deque(sorted(reqs, key=lambda q: q.arrival))
    t = 0.0
    stats = {"iters": 0, "prefill_tokens": 0, "preemptions": 0, "itl": [], "batch": [], "util": []}
    while pending:
        if pending[0].arrival > t:
            t = pending[0].arrival
        batch = []
        while pending and pending[0].arrival <= t and len(batch) < N:
            batch.append(pending.popleft())
        pf = sum(q.prompt for q in batch)
        t += t0 + cp * pf
        stats["iters"] += 1
        stats["prefill_tokens"] += pf
        for q in batch:
            q.first_token = t
            q.generated = 1
            if q.out == 1:
                q.done = t
        for j in range(1, max(q.out for q in batch)):
            dt = t0 + cd * len(batch)          # finished members still occupy their slot
            t += dt
            stats["iters"] += 1
            stats["batch"].append(len(batch))
            live = [q for q in batch if q.generated < q.out]
            stats["itl"].append((dt, len(live)))
            for q in live:
                q.generated += 1
                if q.generated == q.out:
                    q.done = t
    stats["end"] = t
    return stats


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def wpercentile(pairs, p):
    """Weighted percentile of (value, weight) pairs."""
    if not pairs:
        return float("nan")
    v = np.array([x[0] for x in pairs])
    w = np.array([x[1] for x in pairs], dtype=float)
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(v[min(np.searchsorted(c, p / 100.0), len(v) - 1)])


def summarize(reqs, stats, slo_ttft=2.0, slo_tpot=0.08):
    ok = [q for q in reqs if q.done is not None]
    ttft = np.array([q.first_token - q.arrival for q in ok])
    e2e = np.array([q.done - q.arrival for q in ok])
    tpot = np.array([(q.done - q.first_token) / (q.out - 1) for q in ok if q.out > 1])
    toks = sum(q.out for q in ok)
    span = stats["end"] - min(q.arrival for q in reqs)
    good = sum(q.out for q in ok if q.first_token - q.arrival <= slo_ttft and
               (q.out == 1 or (q.done - q.first_token) / (q.out - 1) <= slo_tpot))
    return {
        "tput": toks / span,
        "goodput": good / span,
        "attain": sum(1 for q in ok if q.first_token - q.arrival <= slo_ttft and
                      (q.out == 1 or (q.done - q.first_token) / (q.out - 1) <= slo_tpot)) / len(reqs),
        "ttft50": float(np.percentile(ttft, 50)), "ttft99": float(np.percentile(ttft, 99)),
        "e2e50": float(np.percentile(e2e, 50)), "e2e99": float(np.percentile(e2e, 99)),
        "tpot99": float(np.percentile(tpot, 99)),
        "itl99": wpercentile(stats["itl"], 99),
        "preempt": stats["preemptions"], "rejected": sum(1 for q in reqs if q.rejected),
        "prefill_tokens": stats["prefill_tokens"],
        "util": float(np.mean(stats["util"])) if stats["util"] else float("nan"),
    }


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------
def selftests():
    t0_ = time.time()
    n_ok = 0

    def ok(name):
        nonlocal n_ok
        n_ok += 1
        print(f"  PASS {n_ok:2d}  {name}")

    # 1 closed form for one request
    q = [Req(0, 0.0, 100, 11)]
    simulate_continuous(q)
    exp = T0 + CP * 100 + 10 * (T0 + CD)
    assert abs(q[0].done - exp) < 1e-12, (q[0].done, exp)
    assert abs(q[0].first_token - (T0 + CP * 100)) < 1e-12
    ok("one request: first token and finish times match the closed form")

    # 2 single-token output finishes at the prefill iteration
    q = [Req(0, 1.0, 50, 1)]
    simulate_continuous(q)
    assert q[0].done == q[0].first_token and abs(q[0].done - (1.0 + T0 + CP * 50)) < 1e-12
    ok("a one-token output finishes with its prefill iteration (idle clock jumps to the arrival)")

    # 3 continuous with max_seqs=1 == static N=1, request by request
    w = make_workload(1, 60, 1.0)
    a, b = clone(w), clone(w)
    simulate_continuous(a, max_seqs=1)
    simulate_static(b, 1)
    assert all(abs(x.done - y.done) < 1e-9 and abs(x.first_token - y.first_token) < 1e-9 for x, y in zip(a, b))
    ok("continuous with one slot reproduces static batch-size-1 exactly (60 requests)")

    # 4 every request emits exactly `out` tokens, with and without preemption, many seeds
    pre_total = 0
    for seed in range(12):
        w = make_workload(seed, 120, 6.0, pmed=300, omed=200)
        for nb in (4096, 160):
            c = clone(w)
            st = simulate_continuous(c, n_blocks=nb, debug=True)
            pre_total += st["preemptions"]
            for x in c:
                if not x.rejected:
                    assert x.generated == x.out and x.done is not None
                    assert x.first_token <= x.done and x.arrival <= x.first_token
    assert pre_total > 0
    ok(f"24 runs: every admitted request emits exactly `out` tokens; block accounting invariant holds each iteration ({pre_total} preemptions exercised)")

    # 5 no blocks leak (asserted inside) and the run terminates under heavy pressure
    w = make_workload(3, 150, 20.0, pmed=400, omed=300)
    c = clone(w)
    st = simulate_continuous(c, n_blocks=200, debug=True)
    assert st["preemptions"] > 0 and sum(1 for x in c if x.done) + sum(1 for x in c if x.rejected) == 150
    ok(f"heavy memory pressure: terminates, all blocks returned ({st['preemptions']} preemptions)")

    # 6 a request that can never fit is rejected, not hung
    q = [Req(0, 0.0, 3000, 100), Req(1, 0.0, 100, 10)]
    simulate_continuous(q, n_blocks=100)
    assert q[0].rejected and q[0].done is None and q[1].done is not None
    ok("a request larger than the whole pool is rejected up front; others proceed")

    # 7 prefill tokens equal the sum of prompts when nothing is preempted
    w = make_workload(5, 100, 3.0)
    c = clone(w)
    st = simulate_continuous(c, n_blocks=8192)
    assert st["preemptions"] == 0 and st["prefill_tokens"] == sum(x.prompt for x in c)
    ok("without preemption, total prefill work equals the sum of prompt lengths")

    # 8 with preemption, recompute adds prefill work
    w = make_workload(5, 100, 12.0, pmed=300, omed=300)
    c = clone(w)
    st = simulate_continuous(c, n_blocks=180)
    assert st["preemptions"] > 0 and st["prefill_tokens"] > sum(x.prompt for x in c if not x.rejected)
    ok("with preemption, prefill work exceeds the prompt total (recomputation is real work)")

    # 9 chunked prefill never exceeds the token budget
    w = make_workload(6, 80, 8.0, pmed=600)
    for bud in (64, 256, 1024):
        c = clone(w)
        st = simulate_continuous(c, max_batched_tokens=bud)
        assert all(x.done is not None for x in c)
    c = clone(w)
    st = simulate_continuous(c, max_batched_tokens=100)
    # per-iteration cost is t0 + cp*pf + cd*nd, so pf + nd <= 100 bounds dt
    assert max(dt for dt, _ in st["itl"]) <= T0 + CP * 100 + CD * 100 + 1e-12
    ok("chunked prefill: per-iteration work stays inside max_batched_tokens (bounded step time)")

    # 10 FCFS: without preemption, first tokens come out in arrival order
    w = make_workload(7, 100, 5.0)
    c = clone(w)
    simulate_continuous(c, n_blocks=8192)
    ft = [x.first_token for x in sorted(c, key=lambda x: x.arrival)]
    assert all(a <= b + 1e-12 for a, b in zip(ft, ft[1:]))
    ok("strict FCFS: first-token times are non-decreasing in arrival order (no preemption)")

    # 11 determinism
    a, b = clone(make_workload(8, 100, 5.0)), clone(make_workload(8, 100, 5.0))
    simulate_continuous(a)
    simulate_continuous(b)
    assert [x.done for x in a] == [x.done for x in b]
    ok("same seed, same result")

    # 12 static batching: padding waste is real
    q = [Req(0, 0.0, 50, 2), Req(1, 0.0, 50, 100)]
    st = simulate_static(q, 2)
    assert abs(q[0].done - (T0 + CP * 100 + (T0 + CD * 2))) < 1e-12
    assert abs(q[1].done - (T0 + CP * 100 + 99 * (T0 + CD * 2))) < 1e-9
    ok("static batch: the short request is done early, but the finished slot still costs compute")

    # 13 continuous beats static on the same mixed workload (property, several seeds)
    wins = 0
    for seed in range(5):
        w = make_workload(seed, 300, 6.0)
        a, b = clone(w), clone(w)
        sa = simulate_continuous(a)
        sb = simulate_static(b, 32)
        wins += summarize(a, sa)["tput"] >= summarize(b, sb)["tput"] * 0.999
    assert wins == 5
    ok("continuous batching throughput >= static (N=32) on 5 of 5 seeds")

    # 14 block accounting matches the chapter-43 cache
    try:
        import importlib.util, os
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch43_paged_kv_cache.py")
        spec = importlib.util.spec_from_file_location("ch43", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        r = np.random.default_rng(0)
        for B in (4, 16):
            cache = m.PagedKVCache(500, B, 1, 1)
            cache.add_seq("s")
            z = np.zeros((1, 1), np.float32)
            for L in range(1, 200):
                cache.append("s", z, z)
                assert len(cache.seqs["s"]["table"]) == -(-L // B)
        ok("block count after L tokens equals ceil(L / B), as in the Chapter 43 cache")
    except FileNotFoundError:
        ok("(Chapter 43 file not alongside; cross-check skipped)")

    print(f"ALL PASS   ({time.time() - t0_:.0f} s)  [{n_ok} tests]")


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def agg(vals):
    v = np.array(vals, dtype=float)
    return f"{v.mean():.3g} [{v.min():.3g}-{v.max():.3g}]"


def experiments():
    SEEDS = (0, 1, 2)
    N = 600
    print("assumed cost model: iteration = %.0f ms + %.2f ms per 1000 prefill tokens + %.2f ms per decoding sequence" % (T0 * 1e3, CP * 1e6, CD * 1e3))
    print("SLO: TTFT <= 2 s and mean time per output token <= 80 ms")
    w = make_workload(0, 20000, 1.0)
    print("workload: prompt median 256 (mean %.0f), output median 128 (mean %.0f)" % (np.mean([q.prompt for q in w]), np.mean([q.out for q in w])))
    mo = np.mean([q.out for q in w])
    print("decode-only upper bound with 128 sequences (ignores prefill and ramp): %.0f tokens/s = %.1f requests/s" % (128 / (T0 + 128 * CD), 128 / (T0 + 128 * CD) / mo))

    print("\n=== Experiment 1: static versus continuous batching (3 seeds; continuous pool 2048 blocks = 32768 tokens) ===")
    print(f"{'rate':>5} {'scheme':>14} {'tokens/s':>22} {'goodput':>22} {'TTFT p50':>20} {'TTFT p99':>20} {'attain':>16}")
    for rate in (0.5, 1, 2, 5, 8, 12):
        rows = {}
        for name, fn in (("static N=8", lambda c: simulate_static(c, 8)), ("static N=32", lambda c: simulate_static(c, 32)),
                         ("continuous", lambda c: simulate_continuous(c))):
            acc = []
            for s in SEEDS:
                c = clone(make_workload(s, N, rate))
                acc.append(summarize(c, fn(c)))
            rows[name] = acc
        for name, acc in rows.items():
            print(f"{rate:5.1f} {name:>14} {agg([a['tput'] for a in acc]):>22} {agg([a['goodput'] for a in acc]):>22} "
                  f"{agg([a['ttft50'] for a in acc]):>20} {agg([a['ttft99'] for a in acc]):>20} {agg([a['attain'] for a in acc]):>16}")

    print("\n=== Experiment 2: memory pressure and preemption (rate 8/s, 3 seeds) ===")
    print(f"{'blocks':>7} {'tokens':>8} {'tokens/s':>22} {'preemptions':>16} {'extra prefill %':>16} {'TTFT p99':>20} {'e2e p99':>20} {'rejected':>9}")
    for nb in (4096, 2048, 1024, 512, 256):
        acc = []
        extra = []
        for s in SEEDS:
            c = clone(make_workload(s, N, 8))
            st = simulate_continuous(c, n_blocks=nb)
            sm = summarize(c, st)
            acc.append(sm)
            base = sum(x.prompt for x in c if not x.rejected)
            extra.append(100 * (st["prefill_tokens"] - base) / base)
        print(f"{nb:7d} {nb * 16:8d} {agg([a['tput'] for a in acc]):>22} {agg([a['preempt'] for a in acc]):>16} {agg(extra):>16} "
              f"{agg([a['ttft99'] for a in acc]):>20} {agg([a['e2e99'] for a in acc]):>20} {agg([a['rejected'] for a in acc]):>9}")

    print("\n=== Experiment 3: the watermark (blocks kept free at admission), 512 blocks, rate 8/s ===")
    print(f"{'watermark':>10} {'preemptions':>16} {'tokens/s':>22} {'e2e p99':>20}")
    for wm in (0.0, 0.01, 0.05, 0.10, 0.20):
        acc = []
        for s in SEEDS:
            c = clone(make_workload(s, N, 8))
            st = simulate_continuous(c, n_blocks=512, watermark=wm)
            acc.append(summarize(c, st))
        print(f"{wm:10.2f} {agg([a['preempt'] for a in acc]):>16} {agg([a['tput'] for a in acc]):>22} {agg([a['e2e99'] for a in acc]):>20}")

    print("\n=== Experiment 4: chunked prefill (max_batched_tokens), rate 4/s, long prompts, 3 seeds ===")
    print(f"{'budget':>8} {'ITL p99 (ms)':>22} {'TTFT p50':>20} {'TTFT p99':>20} {'tokens/s':>22} {'attain':>16}")
    for bud in (None, 4096, 1024, 512, 256, 128):
        acc = []
        for s in SEEDS:
            c = clone(make_workload(s, N, 4, pmed=1000, psig=0.6, pmax=4096))
            st = simulate_continuous(c, n_blocks=4096, max_batched_tokens=bud)
            sm = summarize(c, st)
            sm["itl99"] *= 1000
            acc.append(sm)
        print(f"{str(bud):>8} {agg([a['itl99'] for a in acc]):>22} {agg([a['ttft50'] for a in acc]):>20} {agg([a['ttft99'] for a in acc]):>20} "
              f"{agg([a['tput'] for a in acc]):>22} {agg([a['attain'] for a in acc]):>16}")

    print("\n=== Experiment 5: slot cap (max_num_seqs) at rate 12/s, 3 seeds ===")
    print(f"{'max_seqs':>9} {'tokens/s':>22} {'TTFT p99':>20} {'TPOT p99 (ms)':>22} {'attain':>16}")
    for ms in (8, 16, 32, 64, 128, 256):
        acc = []
        for s in SEEDS:
            c = clone(make_workload(s, N, 12))
            sm = summarize(c, simulate_continuous(c, max_seqs=ms, n_blocks=4096))
            sm["tpot99"] *= 1000
            acc.append(sm)
        print(f"{ms:9d} {agg([a['tput'] for a in acc]):>22} {agg([a['ttft99'] for a in acc]):>20} {agg([a['tpot99'] for a in acc]):>22} {agg([a['attain'] for a in acc]):>16}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tests":
        selftests()
    else:
        selftests()
        experiments()
