"""Chapter 35: disaggregated prefill/decode -- transfer maths, hardware mapping, and an interference simulation.

Run:  python3 ch35_disaggregation.py
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


def sim_disagg(reqs, max_batch, cost, link_gbit):
    # prefill instance: single FIFO server
    t = 0.0
    ready = []
    for r in reqs:
        start = max(t, r.arrival)
        t = start + r.prompt / cost.prefill_tps
        first = t                                                     # first token is produced by prefill
        ready.append((t + transfer_ms(r.prompt, link_gbit) / 1e3, r, first))
    # decode instance: iteration-level, never charges prefill time
    pending = sorted(ready, key=lambda x: x[0])
    running, now, i, done = [], 0.0, 0, []
    gen = {}
    while len(done) < len(reqs):
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
    main()
