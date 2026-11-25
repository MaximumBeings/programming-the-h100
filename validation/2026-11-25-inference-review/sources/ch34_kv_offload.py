"""Chapter 34: hierarchical KV-cache offloading -- tier simulator, restore-vs-recompute, and the decode-stall test.

Run:  python3 ch34_kv_offload.py          (the chapter's numbers)
      python3 ch34_kv_offload.py --test   (self-tests)
      python3 ch34_kv_offload.py --extras (four extra experiments)
CPU simulation. Bandwidths are assumed round numbers (parameters). Nothing here is a measurement.
"""
import random
from collections import OrderedDict

# ---- model shape (Llama-3-70B-like) and assumed hardware parameters
LAYERS, KV_HEADS, HEAD_DIM, BYTES = 80, 8, 128, 2
KV_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * BYTES              # 327,680
KV_PER_TOKEN_LAYER = 2 * KV_HEADS * HEAD_DIM * BYTES                 # 4,096 globally
KV_SHARDS = 2                    # evenly sharded GQA heads across the two H100 ranks
KV_PER_TOKEN_LAYER_PER_RANK = KV_PER_TOKEN_LAYER // KV_SHARDS         # 2,048 per rank/link
PCIE_BW = 55e9                   # B/s host<->GPU, PCIe Gen5 x16 practical (assumption)
NVME_BW = 10e9                   # B/s one NVMe drive (assumption; drives range ~7-14 GB/s)
PREFILL_TPS = 20000.0            # prompt tokens/s the cluster recomputes at (assumption; measure yours)
DECODE_STEP_S = 0.021            # s, bandwidth floor from Chapter 32 (weights only)


def restore_seconds(tokens, tier):
    bw = {"cpu": PCIE_BW, "nvme": NVME_BW}[tier]
    return tokens * KV_PER_TOKEN / bw


def recompute_seconds(tokens):
    return tokens / PREFILL_TPS


class TierCache:
    """Exclusive LRU hierarchy of whole sessions: a session lives in exactly one tier, and evictions demote it one
    tier down. cap is in tokens per tier (None = unbounded; a tier of 0 holds nothing and passes sessions down)."""
    def __init__(self, gpu_tok, cpu_tok, nvme_tok):
        self.cap = {"gpu": gpu_tok, "cpu": cpu_tok, "nvme": nvme_tok}
        self.tier = {k: OrderedDict() for k in self.cap}                    # session -> tokens
        self.order = ["gpu", "cpu", "nvme"]

    def used(self, t):
        return sum(self.tier[t].values())

    def _insert(self, t, sid, tokens):
        if self.cap[t] is None:
            self.tier[t][sid] = tokens; return
        if self.cap[t] == 0 or tokens > self.cap[t]:
            nxt = self.order.index(t) + 1
            if nxt < len(self.order): self._insert(self.order[nxt], sid, tokens)
            return
        while self.used(t) + tokens > self.cap[t]:
            vid, vtok = self.tier[t].popitem(last=False)                   # evict LRU, demote it
            nxt = self.order.index(t) + 1
            if nxt < len(self.order): self._insert(self.order[nxt], vid, vtok)
        self.tier[t][sid] = tokens

    def access(self, sid, tokens):
        """Return the tier that held the session ('gpu','cpu','nvme') or None; then place it on the GPU tier."""
        found = None
        for t in self.order:
            if sid in self.tier[t]:
                found = t
                del self.tier[t][sid]
                break
        self._insert("gpu", sid, tokens)
        return found


def run(policy, n_sessions=400, n_requests=6000, tokens=4096, seed=0, zipf=1.1, gpu_sessions=40, cpu_sessions=120):
    gpu_tok = gpu_sessions * tokens
    cpu_tok = {"gpu": 0, "gpu+cpu": cpu_sessions * tokens, "gpu+cpu+nvme": cpu_sessions * tokens}[policy]
    nvme_tok = None if policy == "gpu+cpu+nvme" else 0
    cache = TierCache(gpu_tok, cpu_tok, nvme_tok)
    rng = random.Random(seed)
    weights = [1 / (i + 1) ** zipf for i in range(n_sessions)]              # a few sessions are very popular
    hits = {"gpu": 0, "cpu": 0, "nvme": 0, None: 0}
    lat = 0.0
    for _ in range(n_requests):
        sid = rng.choices(range(n_sessions), weights)[0]
        found = cache.access(sid, tokens)
        hits[found] += 1
        if found in (None,):
            lat += recompute_seconds(tokens)
        elif found in ("cpu", "nvme"):
            lat += min(restore_seconds(tokens, found), recompute_seconds(tokens))   # use whichever is cheaper
    return {k: v / n_requests for k, v in hits.items()}, lat / n_requests


def run_tests():
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    def invariants(c):
        seen = {}
        for t in c.order:
            for sid in c.tier[t]:
                seen[sid] = seen.get(sid, 0) + 1
        within = all(c.cap[t] is None or c.used(t) <= c.cap[t] for t in c.order)
        return all(v == 1 for v in seen.values()), within

    check("KV bytes per token is 327,680 and per layer x 80 equals it", KV_PER_TOKEN == 327680 and KV_PER_TOKEN_LAYER * LAYERS == KV_PER_TOKEN)
    check("restore time is linear in tokens", abs(restore_seconds(2000, "cpu") - 2 * restore_seconds(1000, "cpu")) < 1e-12)
    check("NVMe restore is slower than CPU restore (bandwidth ordering)", restore_seconds(1000, "nvme") > restore_seconds(1000, "cpu"))
    be = 1 / (KV_PER_TOKEN / NVME_BW)
    check("break-even: at prefill = NVMe tokens/s, restore equals recompute", abs(restore_seconds(1000, "nvme") - 1000 / be) < 1e-12)
    # tier mechanics on a hand case: GPU holds 1 session, CPU 1, NVMe unbounded
    c = TierCache(100, 100, None)
    r1, r2, r3 = c.access("a", 100), c.access("b", 100), c.access("c", 100)
    check("first access of a session is a miss (None)", (r1, r2, r3) == (None, None, None))
    check("cascade: c on GPU, b demoted to CPU, a demoted to NVMe", ("c" in c.tier["gpu"], "b" in c.tier["cpu"], "a" in c.tier["nvme"]) == (True, True, True))
    check("access returns the tier that held it, then it moves to GPU", c.access("a", 100) == "nvme" and "a" in c.tier["gpu"] and "c" in c.tier["cpu"])
    check("immediate re-access hits the GPU", c.access("a", 100) == "gpu")
    # LRU order inside a tier
    c = TierCache(200, 400, 0)
    for sid in "abc": c.access(sid, 100)
    check("LRU: with room for 2 on GPU, the oldest (a) is the one demoted", "a" in c.tier["cpu"] and set(c.tier["gpu"]) == {"b", "c"})
    c.access("b", 100); c.access("d", 100)
    check("LRU: touching b makes c the next victim", "c" in c.tier["cpu"] and set(c.tier["gpu"]) == {"b", "d"})
    # oversize sessions
    c = TierCache(100, 100, 0)
    c.access("big", 500)
    check("a session larger than every bounded tier is dropped, so the next access misses", c.access("big", 500) is None)
    c = TierCache(100, 1000, 0)
    c.access("mid", 500)
    check("a session larger than the GPU tier skips to the CPU tier", "mid" in c.tier["cpu"] and c.access("mid", 500) == "cpu")
    # random-walk invariants
    rng = random.Random(1)
    c = TierCache(5 * 64, 8 * 64, 12 * 64)
    ok1 = ok2 = True
    for _ in range(3000):
        c.access(rng.randrange(60), 64)
        a, b = invariants(c)
        ok1 &= a; ok2 &= b
    check("3000 random accesses: a session is never in two tiers", ok1)
    check("3000 random accesses: no bounded tier exceeds its capacity", ok2)
    # nothing is lost while NVMe is unbounded
    c = TierCache(5 * 64, 8 * 64, None)
    ids = set()
    for _ in range(2000):
        sid = rng.randrange(60); ids.add(sid); c.access(sid, 64)
    held = {sid for t in c.order for sid in c.tier[t]}
    check("with unbounded NVMe, every session ever seen is still held somewhere", held == ids)
    # whole-run properties
    runs = {pol: run(pol) for pol in ("gpu", "gpu+cpu", "gpu+cpu+nvme")}
    check("hit fractions sum to 1", all(abs(sum(h.values()) - 1) < 1e-9 for h, _ in runs.values()))
    g = [runs[p][0]["gpu"] for p in runs]
    check("GPU hit rate does not depend on the lower tiers (exclusive LRU)", max(g) - min(g) < 1e-12)
    m = [runs[p][0][None] for p in runs]
    lat = [runs[p][1] for p in runs]
    check("adding a tier never raises the miss rate or the mean resume latency", m[0] >= m[1] >= m[2] and lat[0] >= lat[1] >= lat[2])
    check("determinism: same seed same result, different seed differs", run("gpu+cpu", seed=3) == run("gpu+cpu", seed=3) and run("gpu+cpu", seed=3) != run("gpu+cpu", seed=4))
    gh = [run("gpu", zipf=z)[0]["gpu"] for z in (0.5, 1.1, 1.5)]
    check("more popularity skew means a higher GPU hit rate", gh[0] < gh[1] < gh[2])
    hidden = (DECODE_STEP_S / LAYERS) * PCIE_BW / KV_PER_TOKEN_LAYER_PER_RANK
    check("decode-stall limit: at the printed token count the fetch time equals the per-layer budget",
          abs(hidden * KV_PER_TOKEN_LAYER_PER_RANK / PCIE_BW - DECODE_STEP_S / LAYERS) < 1e-12 and 7000 < hidden < 7100)
    global PREFILL_TPS
    saved, PREFILL_TPS = PREFILL_TPS, 500000.0                  # prefill so fast that recomputing beats any restore
    h_fast, lat_fast = run("gpu+cpu+nvme")
    cap = (1 - h_fast["gpu"]) * recompute_seconds(4096)
    PREFILL_TPS = saved
    check("with very fast prefill the resume cost is capped by recompute (the min() in run)", lat_fast <= cap + 1e-12)
    try:
        run("gpu", n_requests=0); raised = False
    except ZeroDivisionError:
        raised = True
    check("zero requests raises ZeroDivisionError (documented failure)", raised)
    for name, good in results:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
    print(f"{sum(g for _, g in results)}/{len(results)} self-tests pass")
    return all(g for _, g in results)


def extras():
    print("Experiment A: popularity skew (gpu+cpu+nvme, 400 sessions, 6000 requests, seed 0)")
    print("zipf  gpu-hit  cpu-hit  nvme-hit  miss   mean resume latency")
    for z in (0.6, 0.8, 1.1, 1.4):
        h, lat = run("gpu+cpu+nvme", zipf=z)
        print(f"{z:4.1f}  {h['gpu']:7.3f}  {h['cpu']:7.3f}  {h['nvme']:8.3f}  {h[None]:5.3f}  {lat*1e3:8.2f} ms")
    print("\nExperiment B: CPU tier size (gpu+cpu, GPU 40 sessions)")
    print("cpu sessions  gpu-hit  cpu-hit  miss   mean resume latency")
    for cs in (0, 40, 120, 240, 360):
        h, lat = run("gpu+cpu", cpu_sessions=cs)
        print(f"{cs:12d}  {h['gpu']:7.3f}  {h['cpu']:7.3f}  {h[None]:5.3f}  {lat*1e3:8.2f} ms")
    print("\nExperiment C: break-even against prefill speed, 4096-token session")
    print("prefill tok/s  recompute   restore CPU   restore NVMe   cheapest")
    for tps in (5000, 10000, 20000, 30000, 50000, 100000, 200000):
        rc, rcpu, rnv = 4096 / tps, restore_seconds(4096, "cpu"), restore_seconds(4096, "nvme")
        best = min((rc, "recompute"), (rcpu, "CPU restore"), (rnv, "NVMe restore"))[1]
        print(f"{tps:13d}  {rc*1e3:8.1f} ms  {rcpu*1e3:9.1f} ms  {rnv*1e3:10.1f} ms   {best}")
    print("\nExperiment D: seed spread of mean resume latency (gpu+cpu)")
    v = [run("gpu+cpu", seed=s)[1] * 1e3 for s in range(8)]
    print("ms per seed:", " ".join(f"{x:.1f}" for x in v), f" (min {min(v):.1f}, max {max(v):.1f})")


def main():
    print(f"KV bytes/token = {KV_PER_TOKEN:,} ; per token per layer = {KV_PER_TOKEN_LAYER:,}")
    print("\n== restore vs recompute, per context length (assumed: PCIe 55 GB/s, NVMe 10 GB/s, prefill 20k tok/s) ==")
    print("tokens   KV size    restore(CPU)  restore(NVMe)  recompute")
    for n in (1024, 4096, 16384, 65536):
        print(f"{n:6d}  {n*KV_PER_TOKEN/1e9:7.2f} GB  {restore_seconds(n,'cpu')*1e3:9.1f} ms  "
              f"{restore_seconds(n,'nvme')*1e3:10.1f} ms  {recompute_seconds(n)*1e3:8.1f} ms")
    per_tok_cpu = KV_PER_TOKEN / PCIE_BW
    per_tok_nvme = KV_PER_TOKEN / NVME_BW
    print(f"per token: restore CPU {per_tok_cpu*1e6:.1f} us, NVMe {per_tok_nvme*1e6:.1f} us, recompute {1e6/PREFILL_TPS:.1f} us")
    print(f"NVMe restore beats recompute only if prefill speed < {1/per_tok_nvme:,.0f} tok/s; "
          f"CPU restore only if prefill speed < {1/per_tok_cpu:,.0f} tok/s")

    print("\n== tier simulation: 400 sessions, 4096-token contexts, Zipf popularity, GPU holds 40 sessions ==")
    print("policy            gpu-hit  cpu-hit  nvme-hit  miss(recompute)  mean resume latency")
    for pol in ("gpu", "gpu+cpu", "gpu+cpu+nvme"):
        h, lat = run(pol)
        print(f"{pol:16s}  {h['gpu']:7.3f}  {h['cpu']:7.3f}  {h['nvme']:8.3f}  {h[None]:15.3f}  {lat*1e3:9.2f} ms")

    print("\n== can an ACTIVE sequence's KV be paged in during decode without stalling? ==")
    per_layer_compute = DECODE_STEP_S / LAYERS
    print(f"assumed per-layer overlap window in a decode step ~ {per_layer_compute*1e3:.3f} ms (21 ms / 80 layers)")
    for ctx in (1024, 4096, 8192, 32768):
        need = ctx * KV_PER_TOKEN_LAYER_PER_RANK / PCIE_BW
        print(f"context {ctx:6d}: fetching ONE rank's layer KV over its PCIe link takes {need*1e3:6.3f} ms -> "
              f"{'fits ideal overlap window' if need <= per_layer_compute else 'exceeds ideal overlap window'}")
    hidden_tokens = per_layer_compute * PCIE_BW / KV_PER_TOKEN_LAYER_PER_RANK
    print(f"ideal per-rank threshold with TP2 KV-head sharding (one sequence): ~{hidden_tokens:,.0f} tokens")


if __name__ == "__main__":
    import sys
    if "--extras" in sys.argv:
        extras(); sys.exit(0)
    if "--test" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    main()
