# Chapter 32: Inference Engine Anatomy and the Measurement Harness

!!! abstract "Goal and prerequisites"

    Learn what limits an LLM serving engine on a Hopper GPU, how continuous batching works, and how to measure a server honestly. You build a workload generator, a simulated engine and a metrics module that every later project reuses. Read [Chapter 1](01-h100-architecture.md) (memory hierarchy and bandwidth) and [Chapter 14](14-nsight-systems.md) (timelines) first.

!!! warning "Evidence level"

    The harness and engine below run on a **CPU**. The engine is a **simulation** whose cost constants are illustrative parameters, not H100 measurements. What the run verifies: the metric code (against a hand-computed case), the byte and bandwidth arithmetic, and the simulator's internal consistency (Little's law). What it cannot tell you: how fast any real server is. Replace the simulated engine with a client that talks to a real server, keep the rest.

## 32.1 Two phases, two bottlenecks

Generating a reply has two phases.

**Prefill** processes the whole prompt at once. It is a large matrix multiply and is limited by tensor-core throughput. **Decode** produces one token per step. Each step reads every weight and the whole KV cache, does little arithmetic with them, and is limited by memory bandwidth.

```
 prompt (512 tokens)                       reply (128 tokens)
 |-------- prefill --------|--d--|--d--|--d--|--d--| ... |--d--|
        compute-bound        each d = read all weights + KV cache: memory-bound
        sets TTFT            together set TPOT
```

### The decode floor

At batch size 1, one decode step must read every weight once. A lower bound on its time is bytes divided by bandwidth. With the datasheet HBM3 bandwidth of about 3.35 TB/s per H100 SXM:

```
KV bytes/token (Llama-3-70B-shaped, BF16) = 327680 (320 KiB); 4096-token prompt = 1.34 GB
decode floor, 70B BF16 on 2 H100: 20.9 ms/token at batch 1 (weights only; ignores KV reads, all-reduce and launch cost)
decode floor, 70B FP8  on 1 H100: 20.9 ms/token at batch 1
```

The two lines give the same floor because halving the bytes per weight on half the GPUs reads the same bytes per GPU. A 70B model in FP8 is 70 GB of weights, which leaves almost no room for KV cache on an 80 GB card, so treat the single-GPU line as an arithmetic illustration, not a deployment plan.

Check the KV figure against your own model's configuration: it is `2 x layers x kv_heads x head_dim x bytes`, here `2 x 80 x 8 x 128 x 2`.

## 32.2 Continuous batching

Early servers batched requests, ran them together to completion, then took the next batch. A short request waited for the longest one. **Continuous batching** schedules at every decode iteration: finished requests leave, waiting requests join, and the batch stays full.

Batching helps because the weight read is shared by every sequence in the step. The KV cache is not shared, so its cost grows with the batch. This is why throughput rises with load and then flattens.

!!! warning "Common mistake"

    Reporting throughput without latency. A server can show high tokens per second by queueing requests until the batch is large, while every user waits. Throughput and latency must be reported together, at a stated load.

## 32.3 The metrics

| Metric | Definition | What it reveals |
|---|---|---|
| TTFT | Time from arrival to the first token | Queueing plus prefill |
| TPOT | Time per token after the first | Decode speed and batch size |
| p50, p99 | Median and 99th percentile of each | Typical versus worst-case users |
| Throughput | Output tokens per second, whole server | Capacity |
| Goodput | Requests per second that meet **both** SLOs | Useful capacity |
| Cost | `GPU $/hour x GPUs x hours / million tokens` | Unit economics |

### Hand-check the metric code

Two requests: the first arrives at 0.0 s, gets its first token at 0.5 s and finishes at 0.9 s after 5 tokens; the second arrives at 1.0 s, first token at 1.2 s, finishes at 1.4 s after 3 tokens. TTFTs are 0.5 and 0.2, median 0.35. TPOTs are `0.4/4 = 0.10` and `0.2/2 = 0.10`. With a TTFT SLO of 0.4 s only the second request qualifies, so goodput is `1 / 1.4 = 0.714` requests per second.

```
hand-check: ttft_p50=0.350 (expect 0.350)  tpot_p50=0.100 (expect 0.100)  goodput=0.714 req/s (expect 0.714: only req 1 meets TTFT<=0.4)
```

## 32.4 The workload

A fair test fixes the workload and publishes it:

- **Arrivals:** Poisson with a stated rate (exponential gaps), so bursts happen as they do in practice.
- **Lengths:** log-normal prompt and output lengths with stated medians, or a recorded dataset such as a public chat corpus.
- **Sweep:** repeat at several rates, from idle to overloaded.
- **Seed:** fixed, so the run can be repeated.

## 32.5 The simulated engine

The simulator advances one iteration at a time. Each iteration admits waiting requests up to a batch limit, charges prefill time for the newly admitted prompts, then charges a decode step whose cost grows with the number of running sequences and the total cached tokens:

```python
step_t = cost.base + cost.per_seq * len(running) + cost.per_kv_token * kv
dt = prefill_t + step_t
```

Its constants (a 20 ms base step, 0.2 ms per sequence, 40,000 prompt tokens per second) are placeholders chosen to give plausible shapes.

### Check the simulator: Little's law

For any stable system, the average number of requests in the system equals the arrival rate times the average time each spends there (`L = lambda x W`). The simulator measures `L` independently, as the time-average of how many requests are present, and `W` from the per-request times:

```
Little's law: L=14.060  lambda*W=14.105  rel.diff=0.321%
```

Agreement within a fraction of a percent says the bookkeeping is consistent. It does not say the cost model is realistic.

### The load sweep

```
rate(req/s)  ttft_p50  ttft_p99  tpot_p50  tput(tok/s)  slo_ok  $/Mtok(@$4/GPU-h, 2 GPUs)
         1     0.046     0.082    0.0211          158    1.00     14.05
         4     0.051     0.099    0.0252          574    1.00      3.87
         8     0.061     0.121    0.0324         1009    1.00      2.20
        12     2.125     3.507    0.0388         1189    0.49      1.87
        16     4.176     8.611    0.0390         1225    0.37      1.81
```

Read the table as a shape, not as numbers to quote:

- Throughput climbs with load and flattens near 1,200 tokens per second: the batch is full.
- Cost per million tokens falls with load, because the fixed GPU bill is spread over more tokens.
- Between 8 and 12 requests per second TTFT jumps from about 0.06 s to over 2 s and SLO attainment drops to 0.49. The server is past its knee: new requests queue.
- The cheapest point (16 req/s) is also the one where only 37 percent of requests meet the SLO. Cost per token alone is a misleading target; goodput per dollar is better.

The $4 per GPU-hour is an assumed price for the arithmetic, not a quote.

## 32.6 Reusing the harness

| Project | What it plugs in |
|---|---|
| MoE expert parallelism (Ch. 33) | Layer-level communication cost in the step time |
| KV-cache offloading | Extra latency when a session resumes from a slower tier |
| Disaggregation | Separate prefill and decode pools with a transfer cost |
| Benchmark teardown | Real engines behind the same client |
| Autoscaler, router, cache | Cost and SLO attainment at fluctuating load |

For a real server, replace `simulate` with a client that sends the same requests and records first-token and last-token timestamps, and keep `summarize` unchanged.

## Chapter Summary

Serving splits into compute-bound prefill, which sets TTFT, and memory-bound decode, which sets TPOT. The decode floor is weight bytes over bandwidth, about 21 ms per token for a 70B BF16 model across two H100s at batch 1, before KV reads and communication. Continuous batching refills the batch every iteration, and shared weight reads make throughput rise with load until the batch is full. A fair measurement fixes and publishes the workload, reports TTFT, TPOT, percentiles, goodput under an SLO and cost, and sweeps the load. The simulator in this chapter reproduces those shapes and passes a hand-computed metric check and Little's law, but its numbers are illustrative.

## Self-Check Questions

1. Why is decode limited by memory bandwidth while prefill is not?
2. Why can a 70B BF16 model not decode faster than about 21 ms per token on two H100s at batch 1, whatever the software?
3. In the sweep, why does cost per million tokens fall as the rate rises, and why is the cheapest row not the best operating point?
4. Why must a throughput figure always come with a latency figure?
5. What does Little's law check in the simulator, and what does it not check?

## Worked Solutions

1. Prefill multiplies large matrices (many tokens against the weights), so it is arithmetic-heavy. Decode handles one token per sequence and must read all weights and the KV cache for little arithmetic, so time is set by bytes moved.
2. Each step must read every weight once, so time is at least weight bytes divided by aggregate memory bandwidth: `140 GB / (2 x 3.35 TB/s) = 20.9 ms`. Real systems add KV reads, communication and launch overhead.
3. The GPU bill is fixed per hour, so more tokens per hour means a lower cost per token. At 16 requests per second the server is overloaded, so only 37 percent of requests meet the SLO; the useful operating point is the highest load that still meets it.
4. A server can raise throughput by letting requests queue into large batches, which makes users wait. Throughput alone hides that.
5. It checks that the simulator's time-average occupancy equals arrival rate times mean residence time, which catches bookkeeping errors. It does not check that the cost constants resemble a real GPU.
