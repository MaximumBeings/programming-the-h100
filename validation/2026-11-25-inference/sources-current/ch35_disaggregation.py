"""Chapter 35: disaggregated prefill/decode -- transfer maths, hardware mapping, and an interference simulation.

Run:  python3 ch35_disaggregation.py          (the chapter's numbers)
      python3 ch35_disaggregation.py --test   (self-tests)
      python3 ch35_disaggregation.py --extras (four extra experiments)
CPU simulation with the placeholder cost model of Chapter 32. Hardware figures are datasheet values;
prices are ASSUMPTIONS. Nothing here is a measurement of a real cluster.
"""
import math
from ch32_inference_harness import (CostModel, Req, kv_bytes_per_token, make_workload, summarize)

KV = kv_bytes_per_token(80, 8, 128)           # 327,680 B/token for a Llama-3-70B-shaped model


# ------------------------------------------------------------- 1. KV transfer arithmetic
def transfer_ms(tokens, gbit_per_s):
    return tokens * KV / (gbit_per_s * 1e9 / 8) * 1e3


def exposed_ms(tokens, gbit_per_s, prefill_tps):
    """If KV is streamed layer by layer WHILE prefill runs, only the part not hidden behind prefill is exposed
    (best case: perfect overlap)."""
    return max(0.0, transfer_ms(tokens, gbit_per_s) - tokens / prefill_tps * 1e3)


# ------------------------------------------------------------- 2. hardware mapping (datasheet values, check yours)
GPUS = {  # name: (HBM/GDDR bandwidth B/s, memory bytes)
    "H100 SXM": (3.35e12, 80e9),
    "L40S": (0.864e12, 48e9),
}


def decode_config(name, n, weight_bytes=70e9, ctx_tokens=4096):
    bw, mem = GPUS[name]
    floor_ms = weight_bytes / (n * bw) * 1e3                         # weights read once per step
    kv_room = n * mem - weight_bytes
    seqs = max(0, int(kv_room // (ctx_tokens * KV)))                 # ignores activations/overheads: upper bound
    return floor_ms, kv_room / 1e9, seqs


# ------------------------------------------------------------- 3. interference simulation
def sim_colocated(reqs, replicas, max_batch, cost):
    from ch32_inference_harness import simulate
    shards = [[] for _ in range(replicas)]
    for i, r in enumerate(reqs):
        shards[i % replicas].append(Req(len(shards[i % replicas]), r.arrival, r.prompt, r.output))
    out, wall = [], 0.0
    for sh in shards:
        done, w, _ = simulate(sh, max_batch=max_batch, cost=cost)
        out += done; wall = max(wall, w)
    return out, wall


def _decode_instance(pending, max_batch, cost):
    """One decode instance. pending: list of (ready_time, request, first_token_time). Never charges prefill time."""
    pending = sorted(pending, key=lambda x: x[0])
    running, now, i, done = [], 0.0, 0, []
    gen = {}
    while len(done) < len(pending):
        while i < len(pending) and pending[i][0] <= now and len(running) < max_batch:
            running.append(pending[i]); gen[pending[i][1].rid] = 1; i += 1
        if not running:
            now = pending[i][0]; continue
        kv = sum(r.prompt + gen[r.rid] for _, r, _ in running)
        now += cost.base + cost.per_seq * len(running) + cost.per_kv_token * kv
        for item in list(running):
            _, r, first = item
            gen[r.rid] += 1
            if gen[r.rid] >= r.output:
                r.first_token, r.finish = first, now
                done.append(r); running.remove(item)
    return done, now


def sim_disagg_pool(reqs, n_prefill, n_decode, max_batch, cost, link_gbit):
    """n_prefill FIFO prefill servers (a request goes to whichever frees first), then a KV transfer, then one of
    n_decode decode instances (assigned round-robin in order of readiness). Works on copies of the requests."""
    import heapq
    reqs = [Req(r.rid, r.arrival, r.prompt, r.output) for r in reqs]
    free = [0.0] * n_prefill
    ready, finished = [], []
    for r in reqs:
        start = max(heapq.heappop(free), r.arrival)
        t = start + r.prompt / cost.prefill_tps
        heapq.heappush(free, t)
        if r.output <= 1:                                             # nothing left to decode: done at the first token
            r.first_token = r.finish = t
            finished.append(r)
            continue
        ready.append((t + transfer_ms(r.prompt, link_gbit) / 1e3, r, t))   # first token is produced by prefill
    ready.sort(key=lambda x: x[0])
    done, wall = list(finished), max([r.finish for r in finished], default=0.0)
    for k in range(n_decode):
        d, now = _decode_instance(ready[k::n_decode], max_batch, cost)
        done += d; wall = max(wall, now)
    return done, wall


def sim_disagg(reqs, max_batch, cost, link_gbit):
    return sim_disagg_pool(reqs, 1, 1, max_batch, cost, link_gbit)


def _wl(rate, seed=3, n=500, prompt_med=2048, out_med=128):
    return make_workload(n, rate=rate, seed=seed, prompt_med=prompt_med, out_med=out_med)


def run_tests():
    from ch32_inference_harness import simulate
    results = []
    cost = CostModel()

    def check(name, cond):
        results.append((name, bool(cond)))

    check("KV bytes per token is 327,680", KV == 327680)
    check("transfer: 4096 tokens at 400 Gb/s is 26.8 ms (chapter figure)", abs(transfer_ms(4096, 400) - 26.8435) < 0.001)
    check("transfer is linear in tokens and inverse in bandwidth",
          abs(transfer_ms(2000, 100) - 2 * transfer_ms(1000, 100)) < 1e-9 and abs(transfer_ms(1000, 200) * 2 - transfer_ms(1000, 100)) < 1e-9)
    check("exposed is 0 when prefill is slower than the transfer, positive when prefill is very fast",
          exposed_ms(4096, 400, 20000) == 0.0 and exposed_ms(4096, 400, 1e6) > 0)
    check("exposed never exceeds the full transfer time", all(0 <= exposed_ms(n, g, t) <= transfer_ms(n, g) + 1e-9
                                                              for n in (512, 4096, 65536) for g in (100, 400) for t in (1e3, 2e4, 1e6)))
    f2, room2, s2 = decode_config("H100 SXM", 2)
    f4, room4, s4 = decode_config("H100 SXM", 4)
    check("decode floor halves when the GPU count doubles", abs(f2 / 2 - f4) < 1e-9)
    check("chapter figures: 2xH100 10.4 ms / 90.0 GB / 67 seqs; 4xL40S 20.3 ms / 122.0 GB / 90 seqs",
          (round(f2, 1), round(room2, 1), s2) == (10.4, 90.0, 67) and (round(decode_config("L40S", 4)[0], 1), round(decode_config("L40S", 4)[1], 1), decode_config("L40S", 4)[2]) == (20.3, 122.0, 90))
    f1, room1, s1 = decode_config("L40S", 1)
    check("weights that do not fit (70 GB on one 48 GB card): KV room negative, sequences 0, floor still printed (documented failure)",
          room1 < 0 and s1 == 0 and f1 > 0)
    # colocated
    reqs = _wl(6)
    c1, w1 = sim_colocated(reqs, 1, 64, cost)
    ref, wref, _ = simulate(_wl(6), max_batch=64, cost=cost)
    check("one colocated replica equals the plain simulator", sorted((r.first_token, r.finish) for r in c1) == sorted((r.first_token, r.finish) for r in ref))
    c2, _ = sim_colocated(reqs, 2, 64, cost)
    check("colocated x2 returns every request exactly once", sorted(r.arrival for r in c2) == sorted(r.arrival for r in reqs))
    # disaggregated
    before = [(r.first_token, r.finish) for r in reqs]
    d, wd = sim_disagg(reqs, 64, cost, 400)
    check("disagg does not modify its input", before == [(r.first_token, r.finish) for r in reqs])
    check("disagg returns every request exactly once", len(d) == len(reqs) and sorted(r.arrival for r in d) == sorted(r.arrival for r in reqs))
    check("TTFT is at least the request's own prefill time", all(r.first_token - r.arrival >= r.prompt / cost.prefill_tps - 1e-9 for r in d))
    check("decode cannot start before the KV arrives: finish - first >= transfer time",
          all(r.finish - r.first_token >= transfer_ms(r.prompt, 400) / 1e3 - 1e-9 for r in d if r.output > 1))
    byarr = sorted(d, key=lambda r: r.arrival)
    check("single prefill server is FIFO: first tokens come in arrival order", all(a.first_token <= b.first_token for a, b in zip(byarr, byarr[1:])))
    check("determinism", [(r.first_token, r.finish) for r in sim_disagg(_wl(10), 64, cost, 400)[0]] == [(r.first_token, r.finish) for r in sim_disagg(_wl(10), 64, cost, 400)[0]])
    slow = summarize(sim_disagg(_wl(10), 64, cost, 100)[0], 1.0)["tpot_p50"]
    fast = summarize(sim_disagg(_wl(10), 64, cost, 400)[0], 1.0)["tpot_p50"]
    free = summarize(sim_disagg(_wl(10), 64, cost, 1e9)[0], 1.0)["tpot_p50"]
    check("a faster link never raises median TPOT (100 Gb/s >= 400 Gb/s >= ~free)", slow >= fast >= free)
    one = [Req(0, 0.0, 1000, 1)]
    do, _ = sim_disagg(one, 64, cost, 400)
    check("a one-token reply finishes at its first token (no extra decode step)", do[0].finish == do[0].first_token)
    # exact single-request case: prompt 1000, output 3, arrival 0
    one3 = sim_disagg([Req(0, 0.0, 1000, 3)], 64, cost, 400)[0][0]
    t_pf = 1000 / cost.prefill_tps
    ready_t = t_pf + transfer_ms(1000, 400) / 1e3
    step = lambda kv: cost.base + cost.per_seq * 1 + cost.per_kv_token * kv
    expect = ready_t + step(1000 + 1) + step(1000 + 2)
    check("hand case: one request, prompt 1000, output 3: first = prefill time, finish = prefill + transfer + 2 decode steps",
          abs(one3.first_token - t_pf) < 1e-12 and abs(one3.finish - expect) < 1e-12)
    one3_slow = sim_disagg([Req(0, 0.0, 1000, 3)], 64, cost, 100)[0][0]
    check("hand case: a 4x slower link delays that request's finish by exactly the extra transfer time",
          abs((one3_slow.finish - one3.finish) - (transfer_ms(1000, 100) - transfer_ms(1000, 400)) / 1e3) < 1e-12)
    pool, _ = sim_disagg_pool(_wl(14), 1, 1, 64, cost, 400)
    base, _ = sim_disagg(_wl(14), 64, cost, 400)
    check("pool(1,1) equals the original 1+1 function", sorted((r.first_token, r.finish) for r in pool) == sorted((r.first_token, r.finish) for r in base))
    p2, _ = sim_disagg_pool(_wl(14), 2, 2, 64, cost, 400)
    check("2+2 completes every request and total generated work equals 1+1", len(p2) == 500 and sum(r.output for r in p2) == sum(r.output for r in base))
    # the chapter's three claims, as regression tests on the published seed
    m = {}
    for rate in (6, 10, 14):
        co, wc = sim_colocated(_wl(rate), 2, 64, cost)
        di, wdd = sim_disagg(_wl(rate), 64, cost, 400)
        m[rate] = (summarize(co, wc), summarize(di, wdd))
    check("claim 1: at 6 and 10 req/s the split has lower TPOT p99", all(m[r][1]["tpot_p99"] < m[r][0]["tpot_p99"] for r in (6, 10)))
    check("claim 2: at 6 req/s the split has higher TTFT p99", m[6][1]["ttft_p99"] > m[6][0]["ttft_p99"])
    check("claim 3: at 14 req/s the split's TPOT p99 is worse and its SLO attainment lower",
          m[14][1]["tpot_p99"] > m[14][0]["tpot_p99"] and m[14][1]["slo_attainment"] < m[14][0]["slo_attainment"])
    for name, good in results:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
    print(f"{sum(g for _, g in results)}/{len(results)} self-tests pass")
    return all(g for _, g in results)


def extras():
    cost = CostModel()
    print("Experiment A: pool ratio at fixed GPU count (500 requests, prompt median 2048, output median 128, seed 3)")
    print("GPUs  rate  layout        ttft_p99  tpot_p99  slo_ok")
    for gpus, rate in ((3, 21), (4, 28)):
        co, w = sim_colocated(_wl(rate), gpus, 64, cost); m = summarize(co, w)
        print(f"{gpus:4d}  {rate:4d}  colocated x{gpus}  {m['ttft_p99']:8.3f}  {m['tpot_p99']:8.4f}  {m['slo_attainment']:6.2f}")
        for npf in range(1, gpus):
            d, w = sim_disagg_pool(_wl(rate), npf, gpus - npf, 64, cost, 400); m = summarize(d, w)
            print(f"{gpus:4d}  {rate:4d}  disagg {npf}+{gpus-npf}     {m['ttft_p99']:8.3f}  {m['tpot_p99']:8.4f}  {m['slo_attainment']:6.2f}")
    print("\nExperiment B: link speed (1+1, rate 10)")
    print("link Gb/s  ttft_p99  tpot_p50  tpot_p99")
    for g in (25, 100, 200, 400, 1e9):
        d, w = sim_disagg(_wl(10), 64, cost, g); m = summarize(d, w)
        print(f"{g:9.0f}  {m['ttft_p99']:8.3f}  {m['tpot_p50']:8.4f}  {m['tpot_p99']:8.4f}")
    print("\nExperiment C: prompt length at rate 6, 2 GPUs (colocated x2 vs 1+1)")
    print("prompt_med  mode        ttft_p99  tpot_p99  slo_ok")
    for pm in (256, 2048, 8192):
        co, w = sim_colocated(_wl(6, prompt_med=pm), 2, 64, cost); m = summarize(co, w)
        print(f"{pm:10d}  colocated   {m['ttft_p99']:8.3f}  {m['tpot_p99']:8.4f}  {m['slo_attainment']:6.2f}")
        d, w = sim_disagg(_wl(6, prompt_med=pm), 64, cost, 400); m = summarize(d, w)
        print(f"{pm:10d}  disagg 1+1  {m['ttft_p99']:8.3f}  {m['tpot_p99']:8.4f}  {m['slo_attainment']:6.2f}")
    print("\nExperiment D: seed spread at rate 10 (TPOT p99, colocated x2 vs 1+1)")
    wins = 0
    for sd in range(8):
        co, w = sim_colocated(_wl(10, seed=sd), 2, 64, cost); a = summarize(co, w)["tpot_p99"]
        d, w = sim_disagg(_wl(10, seed=sd), 64, cost, 400); b = summarize(d, w)["tpot_p99"]
        wins += b < a
        print(f"seed {sd}: colocated {a:.4f}  disagg {b:.4f}  {'split wins' if b < a else 'colocated wins'}")
    print(f"split has the lower TPOT p99 in {wins} of 8 seeds")


def main():
    print(f"KV bytes/token = {KV:,}")
    print("\n== KV transfer time for one prompt (no overlap) and EXPOSED time with ideal layer-wise streaming ==")
    print("tokens   KV size     100 Gb/s   200 Gb/s   400 Gb/s    exposed@400 (prefill 20k tok/s)")
    for n in (1024, 4096, 16384, 65536):
        print(f"{n:6d}  {n*KV/1e9:7.2f} GB  {transfer_ms(n,100):8.1f}   {transfer_ms(n,200):8.1f}   {transfer_ms(n,400):8.1f} ms  "
              f"{exposed_ms(n,400,20000):8.1f} ms")

    print("\n== decode hardware mapping: 70B FP8 weights (70 GB), 4096-token sequences (datasheet bandwidth/capacity) ==")
    print("config          decode floor    KV room   max 4K-token seqs (upper bound)")
    for name, n in (("H100 SXM", 2), ("L40S", 2), ("L40S", 4)):
        f, room, s = decode_config(name, n)
        print(f"{n} x {name:9s}  {f:8.1f} ms/tok  {room:6.1f} GB   {s:4d}")

    print("\n== interference simulation: same total GPUs (2 instances); placeholder cost model ==")
    cost = CostModel()
    print("rate  mode          ttft_p99   tpot_p99   slo_ok(ttft<=2s,tpot<=0.1s)")
    for rate in (6, 10, 14):
        reqs = make_workload(500, rate=rate, seed=3, prompt_med=2048, out_med=128)
        col, w1 = sim_colocated(reqs, 2, 64, cost)
        m1 = summarize(col, w1)
        dis, w2 = sim_disagg(make_workload(500, rate=rate, seed=3, prompt_med=2048, out_med=128), 64, cost, 400)
        m2 = summarize(dis, w2)
        print(f"{rate:4d}  colocated x2  {m1['ttft_p99']:8.3f}  {m1['tpot_p99']:9.4f}  {m1['slo_attainment']:8.2f}")
        print(f"{rate:4d}  disagg 1+1    {m2['ttft_p99']:8.3f}  {m2['tpot_p99']:9.4f}  {m2['slo_attainment']:8.2f}")


if __name__ == "__main__":
    import sys
    if "--extras" in sys.argv:
        extras(); sys.exit(0)
    if "--test" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    main()
