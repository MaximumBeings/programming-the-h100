# Chapter 34: Hierarchical KV-Cache Offloading

!!! abstract "Goal and prerequisites"

    Learn when moving KV cache out of GPU memory pays off, and when it is slower than recomputing it. You build a tier simulator (GPU, CPU RAM, NVMe) and a rule for deciding what can be fetched without stalling decode. Read [Chapter 32](32-inference-engine-and-harness.md) (KV size and the decode floor) and [Chapter 5](05-cp-async-bulk-l2-hinting.md) (asynchronous copies) first.

!!! warning "Evidence level"

    CPU simulation with **assumed** bandwidths (PCIe 55 GB/s, one NVMe drive 10 GB/s) and an **assumed** prefill speed of 20,000 tokens per second. Change those three numbers to your measurements and the conclusions change with them; that sensitivity is the point of the chapter. Nothing here measures a real offload system.

## 34.1 Why offload at all

The KV cache is the largest per-user object in a serving system: about 320 KiB per token for a Llama-3-70B-shaped model, so a 4,096-token conversation holds 1.34 GB. A GPU has tens of gigabytes to spare, so it holds tens of idle conversations. Chat and agent workloads keep many sessions that pause between turns and come back. Throwing the cache away forces a full prefill when the user returns; keeping all of it on the GPU does not fit.

A tiered design keeps hot sessions on the GPU, warm ones in host RAM and cold ones on NVMe:

```
 GPU HBM          ~3.35 TB/s    smallest, active sequences
 CPU RAM          PCIe ~55 GB/s    larger, recently used sessions
 NVMe SSD         ~10 GB/s       largest, cold sessions
```

## 34.2 The first question: restore or recompute?

Restoring a session means copying its KV back. The alternative is to run prefill on the saved text and rebuild it. Offloading only helps if the copy is faster.

### Worked Example 34.2.1

```
== restore vs recompute, per context length (assumed: PCIe 55 GB/s, NVMe 10 GB/s, prefill 20k tok/s) ==
tokens   KV size    restore(CPU)  restore(NVMe)  recompute
  1024     0.34 GB        6.1 ms        33.6 ms      51.2 ms
  4096     1.34 GB       24.4 ms       134.2 ms     204.8 ms
 16384     5.37 GB       97.6 ms       536.9 ms     819.2 ms
 65536    21.47 GB      390.5 ms      2147.5 ms    3276.8 ms
per token: restore CPU 6.0 us, NVMe 32.8 us, recompute 50.0 us
NVMe restore beats recompute only if prefill speed < 30,518 tok/s; CPU restore only if prefill speed < 167,847 tok/s
```

Everything here scales linearly in tokens, so the comparison is per token. With these assumptions a CPU restore is about eight times faster than recomputing, and an NVMe restore is only about 1.5 times faster.

The last line is the practical rule. **If your prefill is faster than about 30,000 tokens per second, reading from one NVMe drive loses to recomputing.** A cluster with fast prefill, short contexts or a slow drive should not use the NVMe tier at all. This is a number to measure on your own system, not to assume.

## 34.3 A tier simulator

The simulator holds whole sessions in least-recently-used (LRU) order. When a tier is full it demotes its oldest session to the next tier. A request for a session finds it in the first tier that holds it, pays the cheaper of restore and recompute, then moves it to the GPU. Session popularity follows a Zipf distribution: a few sessions are returned to constantly, most rarely.

```python
def access(self, sid, tokens):
    found = None
    for t in self.order:                      # gpu, cpu, nvme
        if sid in self.tier[t]:
            found = t
            del self.tier[t][sid]
            break
    self._insert("gpu", sid, tokens)          # evictions cascade downwards
    return found
```

### Worked Example 34.3.1

400 sessions of 4,096 tokens, 6,000 requests, a GPU that holds 40 sessions and a CPU tier that holds 120:

```
policy            gpu-hit  cpu-hit  nvme-hit  miss(recompute)  mean resume latency
gpu                 0.598    0.000     0.000            0.402      82.40 ms
gpu+cpu             0.598    0.236     0.000            0.166      39.79 ms
gpu+cpu+nvme        0.598    0.236     0.104            0.062      32.42 ms
```

The CPU tier cuts mean resume latency from 82 ms to 40 ms by converting about 24 percent of requests from full recomputes into 24 ms restores. Adding NVMe helps less: 40 ms down to 32 ms. That follows from the previous section: an NVMe restore is 134 ms against a 205 ms recompute, a modest gain for the 10 percent of requests it serves.

The hit rates depend on the popularity skew, the tier sizes and the reuse pattern. Measure them from real traffic traces before sizing tiers.

!!! warning "Where offload wins in practice"

    The reliable wins are **reuse**: multi-turn chat, shared system prompts and documents, and preempted requests that are swapped out and back. Offload does not make a single long conversation cheaper to run. Production systems built around this idea include LMCache and the KV-transfer layers in vLLM and similar engines.

## 34.4 What cannot be offloaded: the active sequence

One might hope to keep a long context in host RAM and stream it in for every decode step. The arithmetic says otherwise. A decode step is bounded by reading weights (21 ms on two H100s, Chapter 32), and with 80 layers that is about 0.26 ms of compute per layer. To hide a fetch behind that, the next layer's KV must arrive within 0.26 ms:

```
per-layer compute budget in a decode step ~ 0.263 ms (21 ms / 80 layers)
context   1024: fetching ONE layer's KV over PCIe takes  0.076 ms -> hidden behind compute
context   4096: fetching ONE layer's KV over PCIe takes  0.305 ms -> STALLS decode
context   8192: fetching ONE layer's KV over PCIe takes  0.610 ms -> STALLS decode
context  32768: fetching ONE layer's KV over PCIe takes  2.440 ms -> STALLS decode
largest per-layer KV fetch that hides behind one layer's compute (one sequence): ~3,525 tokens
```

A single sequence of about 3,500 tokens is the limit for fetching its KV a layer ahead, and a real batch holds many sequences all competing for the same PCIe link. So the rule is:

| Case | Offloadable? |
|---|---|
| Idle session between turns | Yes: restore before the next request, off the critical path |
| Preempted request | Yes: swap out and back |
| Shared prefix reused by many requests | Yes: restore once, serve many |
| Active sequence read every decode step | No: keep on the GPU |
| Long active sequence | Reduce the KV itself (quantized cache, fewer KV heads, sparse or windowed attention) |

"Without stalling decode" is achievable only by restoring **ahead** of need, using pinned host memory and a separate copy stream so the transfer overlaps other requests' compute ([Chapter 14](14-nsight-systems.md) shows how to see whether it does), and never by paging in the sequence being decoded.

## 34.5 What to measure on real hardware (untested)

!!! warning "UNTESTED ON H100"

    | Measurement | How |
    |---|---|
    | Real PCIe and NVMe bandwidth | `cudaMemcpy` from pinned memory; a large sequential read with direct I/O. Do not trust nominal figures |
    | Real prefill speed | Your engine on your model, at your prompt lengths |
    | Break-even | The inequality from Worked Example 34.2.1 with the numbers above |
    | Hit rate per tier | Replay a recorded session trace, not synthetic Zipf |
    | Overlap | Nsight Systems timeline: does the copy stream run beside compute without delaying the decode step? |
    | End effect | TTFT for returning sessions and tokens per second for active ones, using the Chapter 32 harness |

## Chapter Summary

KV cache costs about 320 KiB per token, so idle sessions crowd out active ones, and tiering them into GPU, CPU RAM and NVMe can avoid recomputation. Whether a tier helps depends on one comparison: restoring KV from the tier against re-running prefill. With an assumed 55 GB/s PCIe link, 10 GB/s NVMe and 20,000 tokens per second prefill, a CPU restore beats recompute by about eight times but one NVMe drive by only about 1.5, and NVMe loses outright if prefill exceeds about 30,000 tokens per second. A Zipf-popularity simulation cut mean resume latency from 82 ms to 40 ms with a CPU tier and to 32 ms with NVMe added. The active sequence cannot be paged in each step: one layer's fetch over PCIe exceeds the 0.26 ms per-layer compute budget beyond about 3,500 tokens. Offloading therefore serves idle, preempted and shared-prefix sessions, not the sequence being decoded.

## Self-Check Questions

1. Why is the restore-versus-recompute comparison independent of context length in this model?
2. At what prefill speed does the NVMe tier stop paying for itself, and why does a faster prefill move the answer in that direction?
3. Why does adding NVMe to a CPU tier help less than adding the CPU tier did?
4. Why can't an active sequence's KV be streamed from host memory every decode step?
5. Name two kinds of workload where offloading clearly helps and one where it does not.

## Worked Solutions

1. Both restore time and recompute time grow in proportion to the number of tokens, so the per-token costs (6.0, 32.8 and 50.0 microseconds) decide the question at any length.
2. At about 30,500 tokens per second. Recompute cost per token falls as prefill speeds up, and once it drops below the 32.8 microseconds per token an NVMe read costs, rebuilding is cheaper than reading.
3. The CPU tier restores in 24 ms against a 205 ms recompute, a large gain on a large share of requests. The NVMe tier restores in 134 ms, only modestly faster than recompute, and serves a smaller share.
4. Each layer's KV for a 4,096-token sequence takes about 0.3 ms to fetch over PCIe, longer than the 0.26 ms of compute available per layer, and every sequence in the batch would need the same link at once, so the fetch cannot be hidden.
5. Multi-turn chat or agent sessions that resume after a pause, and shared system prompts or documents reused across requests, benefit. A single long conversation being decoded continuously does not, because its KV is needed every step.
