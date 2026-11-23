"""Chapter 34: hierarchical KV-cache offloading -- tier simulator, restore-vs-recompute, and the decode-stall test.

Run:  python3 ch34_kv_offload.py
CPU simulation. Bandwidths are assumed round numbers (parameters). Nothing here is a measurement.
"""
import random
from collections import OrderedDict

# ---- model shape (Llama-3-70B-like) and assumed hardware parameters
LAYERS, KV_HEADS, HEAD_DIM, BYTES = 80, 8, 128, 2
KV_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * BYTES              # 327,680
KV_PER_TOKEN_LAYER = 2 * KV_HEADS * HEAD_DIM * BYTES                 # 4,096
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
    """Inclusive-by-demotion LRU hierarchy of whole sessions. cap in tokens per tier (None = unbounded)."""
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


def run(policy, n_sessions=400, n_requests=6000, tokens=4096, seed=0, zipf=1.1):
    gpu_tok = 40 * tokens
    cpu_tok = {"gpu": 0, "gpu+cpu": 120 * tokens, "gpu+cpu+nvme": 120 * tokens}[policy]
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
    print(f"per-layer compute budget in a decode step ~ {per_layer_compute*1e3:.3f} ms (21 ms / 80 layers)")
    for ctx in (1024, 4096, 8192, 32768):
        need = ctx * KV_PER_TOKEN_LAYER / PCIE_BW
        print(f"context {ctx:6d}: fetching ONE layer's KV over PCIe takes {need*1e3:6.3f} ms -> "
              f"{'hidden behind compute' if need <= per_layer_compute else 'STALLS decode'}")
    hidden_tokens = per_layer_compute * PCIE_BW / KV_PER_TOKEN_LAYER
    print(f"largest per-layer KV fetch that hides behind one layer's compute (one sequence): ~{hidden_tokens:,.0f} tokens")


if __name__ == "__main__":
    main()
