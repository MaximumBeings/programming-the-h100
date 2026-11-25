# Chapter 32: Inference Engine Anatomy and the Measurement Harness

[![Chapter 32: Measure before tuning](images/chapter-openers/32-inference.svg)](images/chapter-openers/32-inference.svg)


!!! abstract "Goal and prerequisites"

    Learn what limits an LLM serving engine on a Hopper GPU, how continuous batching works, and how to measure a server honestly. You build a workload generator, a simulated engine and a metrics module that every later project reuses. Read [Chapter 1](01-h100-architecture.md) (memory hierarchy and bandwidth) and [Chapter 14](14-nsight-systems.md) (timelines) first.

!!! warning "Evidence level"

    The harness and engine below run on a **CPU**. The engine is a **simulation** whose cost constants are illustrative parameters, not H100 measurements. What the run verifies: the metric code (against a hand-computed case), the byte and bandwidth arithmetic, and the simulator's internal consistency (Little's law). What it cannot tell you: how fast any real server is. Replace the simulated engine with a client that talks to a real server, keep the rest.

## Worked system map

[![Measure before tuning: step-by-step hand trace](images/worked/ch32-inference.svg)](images/worked/ch32-inference.svg)


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

## 32.7 Self-tests (21, all pass)

The metric hand-check and Little's law are two checks. The harness is reused by Chapters 33 to 46, so it gets a wider net. Run `python3 /Users/oluwaseyiawoga/Developer/programming-the-h100/docs/examples/inference-systems/ch32_inference_harness.py --test`:

| Group | What is checked |
|---|---|
| Arithmetic (4) | KV bytes per token is 327,680 for the 70B shape and linear in layers; the decode floor halves when GPUs double and is linear in bytes per parameter |
| Percentile (4) | `q=0` is the minimum and `q=1` the maximum; the median of `[1,2,3,4]` is 2.5; a single element returns itself; the function is monotone in `q` |
| Workload (3) | the same seed gives identical requests and a different seed does not; arrivals never decrease and lengths are at least 1 |
| Engine (4) | `simulate` does not change its input; every request finishes with exactly `output` tokens and ordered timestamps; `max_batch=1` never overlaps two requests; at 10 times the arrival rate with the same lengths, p99 TTFT does not fall |
| Metrics (4) | the hand case (TPOT 0.10, TTFT p50 0.35, goodput 1/1.4); a tighter SLO never raises attainment; a one-token reply does not divide by zero; the cost formula gives $8,000 per million tokens for 1,000 tokens in one hour on two GPUs at $4 |
| Conservation (2) | Little's law within 2 percent on five seeds; `pct([])` raises `IndexError`, a documented failure |

```
21/21 self-tests pass
```

**The tests can fail.** Two deliberate breakages were tried on a copy: removing the batch cap from the admission loop fails the `max_batch=1` test, and dropping the `- 1` from the TPOT denominator fails the hand case. The second one passed all 20 earlier tests, which is why the hand case is part of the suite and not only a printed line.

## 32.8 Experiment A: the batch cap

The engine takes `max_batch`, the most sequences it will run in one iteration. In a real server an explicit scheduler limit such as vLLM’s `max_num_seqs` acts alongside KV-cache capacity and the token budget, so this experiment shows the shape of the trade, not a tuning guide. Rate 12 requests per second, 300 requests, seed 2 (`--extras`):

```
max_batch  ttft_p50  ttft_p99  tpot_p50  tput(tok/s)  slo_ok
        4   119.523   225.917    0.0211          184    0.03
        8    56.060   106.728    0.0223          340    0.04
       16    25.106    47.547    0.0247          585    0.09
       32     8.853    18.507    0.0295          899    0.21
       64     2.125     3.507    0.0388         1189    0.49
      128     0.077     0.160    0.0442         1246    1.00
```

- Throughput rises with the cap and TPOT also rises: each step carries more sequences, so each token takes longer, but more tokens leave per second.
- TTFT falls by three orders of magnitude, because the cap is what keeps arrivals waiting.
- Caps of 96, 112, 128 and 192 give the same result (TTFT p99 0.160 to 0.166 or lower, attainment 1.00). The cap stops mattering once it exceeds the number of sequences that are ever running at once, which here is between 81 and 96.
- The 64 row equals the 12 req/s row of the section 32.5 sweep. That is a consistency check: same seed, same default cap.

## 32.9 Experiment B: prefill interference

In the engine, a newly admitted prompt's prefill time is added to the iteration, so every running sequence waits for it. Rate 4 requests per second, 300 requests, seed 2, output median 128, prompt median varied:

```
prompt_median  ttft_p50  tpot_p50  tpot_p99  tput(tok/s)
          128     0.038    0.0238    0.0250          575
          512     0.051    0.0252    0.0279          574
         2048     0.105    0.0328    0.0458          573
         8192    13.962    0.1335    0.2249          426
```

- From 128 to 2048 the median TPOT rises from 23.8 to 32.8 ms even though the generated output is identical: the decode steps are stalled by other users' prefills.
- The 8192 row breaks the pattern. Throughput falls from about 575 to 426 tokens per second at the same offered rate, and TTFT p50 is 14 seconds. Prefill alone is saturating the server. The mean of a log-normal is `median x exp(sigma^2/2)` = `8192 x 1.2776` ≈ 10,466 tokens, which is 0.262 s of prefill per request at 40,000 tokens per second, and four requests per second need 1.05 s of prefill per second of wall time. Offered load exceeds capacity.
- Real engines mitigate this by splitting a long prefill into chunks and mixing them with decode steps (Chapter 44). The simulator does not, which is why this is the baseline that chapter improves on.

## 32.10 Failure modes and edge cases

| Situation | What happens | Where it shows |
|---|---|---|
| Offered load above capacity | the queue grows without bound during the run; wall time includes draining the tail, so measured throughput falls below offered load | Experiment A at cap 4; Experiment B at 8192 |
| Near the knee, few requests | results depend heavily on the seed: at 12 req/s, five seeds give TTFT p50 of 1.623, 0.762, 2.125, 0.752 and 0.108 s and SLO attainment of 0.58, 1.00, 0.49, 0.85 and 0.97. At 8 req/s the same seeds all give 0.058 to 0.063 s | report several seeds or a confidence interval near the knee |
| p99 from few samples | with 300 requests, p99 is the third-largest value; with 20 it is nearly the maximum | `pct` interpolates, it does not warn |
| Empty request list | `pct` raises `IndexError` | self-test 21 |
| Reply of one token | TPOT is defined as 0 (division guarded) | self-test 18 |
| SLO thresholds chosen after seeing the data | attainment can be made to look good by tuning the SLO | fix the SLO before the run |
| Cost computed from wall time | an overloaded run has a long drain, which raises cost per token for a reason that is not the engine's efficiency | compare at equal attainment |
| Little's law passes | says the bookkeeping is consistent for **any** cost constants, including absurd ones | do not read it as validation of the cost model |

**Not modelled:** KV-cache memory and the preemption it forces (Chapters 43 and 44); chunked prefill; prefix reuse (Chapter 45); more than one GPU, communication, and pipeline bubbles; speculative decoding; request cancellation, timeouts and retries; tokenizer, network and queueing outside the engine; variation in step time (the model uses a constant); GPU clock and thermal behaviour; streaming back-pressure from slow clients.

## 32.11 On real hardware (UNTESTED)

Nothing in this chapter has been run on a GPU server. These steps are from memory and must be checked against the version you install; flag names change between releases.

1. **Pick the engine and record its version.** For example, `python3 -m pip show vllm` (or the version your container reports) and `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv`. Write both into the run record.
2. **Start the server with fixed settings** and write them down: model, tensor-parallel size, maximum sequence length, maximum number of running sequences, GPU memory fraction. A result without these is not comparable.
3. **Replace `simulate` with a client.** It must send the same `Req` list at the same arrival times, record the time of the first streamed token and of the last, and fill `first_token`, `finish` and `generated`. Then call `summarize` unchanged. Use the same `make_workload` seed so the simulated and real runs see identical requests.
4. **Check the decode floor.** Run a single request with a long output and no other load. Measure TPOT. Compare it with `decode_floor_ms` only under the model’s assumptions: dense execution of the stated weight precision, batch 1, all weights read from HBM on each step, and the stated GPU bandwidth. A lower result can indicate changed precision, caching, sparsity, different hardware or a timing/counting error; it does not by itself prove a measurement is wrong.
5. **Sweep the rate** across idle, loaded and overloaded, repeat each point with at least three seeds, and plot TTFT p50 and p99, TPOT p50, throughput and attainment against rate. Find the knee as in section 32.5.
6. **Fit the cost model.** From the single-request run take `base`, from a batch-of-N run take `per_seq`, and rerun the simulator with those constants. Compare the simulated and real knees. The difference is the part of the model this chapter did not capture.
7. **Capture a profile** of one decode iteration with Nsight Systems ([Chapter 14](14-nsight-systems.md)) to confirm the iteration is dominated by the weight read, as section 32.1 assumes.

## 32.12 Programming exercises

1. In Experiment A, which caps among 64, 80, 96 meet at least 95 percent attainment at 12 req/s? What does the cost per million tokens do between them (default 2 GPUs at $4 per hour)?
2. Compute the decode floor for an 8-billion-parameter model on one H100 in BF16 and in FP8, and its KV bytes per token for 32 layers, 8 KV heads and head dimension 128.
3. At 4 req/s, at what prompt median does prefill alone saturate the simulator? Compare with Experiment B.
4. Chunked prefill: change `simulate` so each iteration charges at most `chunk` prompt tokens of prefill and keeps the rest of a prompt waiting for later iterations. State, before running, what happens to TPOT p99 and to TTFT p50 in Experiment B at prompt median 2048.
5. Show that Little's law cannot detect a wrong `per_kv_token` constant, by changing it by a factor of 100 and rerunning the check.
6. Add a test that two runs of `simulate` on the same workload give identical results, and one that scaling all cost constants by 2 does not reduce any request's latency.

### Solutions

1. Caps 80 and 96 both reach attainment 1.00, and 64 does not (0.49). Cost per million tokens: $1.87 at 64, $1.79 at 80, $1.78 at 96. The cap that is cheapest per token is also the one that meets the SLO, because a cap that is too low leaves the GPU idle behind a queue while its fixed cost keeps accruing. Past 96 nothing changes, since the cap is no longer the binding limit.
2. `8e9 x 2 B / 3.35e12 B/s` = 4.78 ms in BF16 and 2.39 ms in FP8 (weights only, batch 1). KV per token: `2 x 32 x 8 x 128 x 2` = 131,072 bytes = 128 KiB.
3. Prefill demand is `rate x median x exp(sigma^2/2) / prefill_tps`. Setting it to 1 with sigma 0.7 gives `median = 40,000 / (4 x 1.2776)` ≈ 7,827 tokens. Decode steps also take time, so the true saturation point is lower. Experiment B shows 2048 healthy (about 26 percent prefill load) and 8192 overloaded (about 105 percent), which is consistent.
4. Prediction, not a measurement: TPOT p99 should fall, because a decode step no longer waits for a whole long prompt, only for one chunk. TTFT p50 should rise slightly, because a prompt is now finished over several iterations. Chapter 44 builds this and tests it.
5. The check compares `area/wall` with `lambda x W`; both are computed from the same simulated timeline. Any per-iteration cost, however wrong, produces a consistent timeline, so the two sides agree. Rerunning with `per_kv_token = 1e-6` (workload of 300 requests at 4 req/s, seed 1) leaves the relative difference at 0.316 percent (0.322 percent at the default), while the mean request latency goes from 3.955 s to 8.165 s. The conclusion in section 32.5 stands: it validates bookkeeping only.
6. For identical runs, compare every `(first_token, finish)` pair. For the scaling test, multiply `base`, `per_seq`, `per_kv_token` by 2 and divide `prefill_tps` by 2; every `finish - arrival` must not decrease. This held for all 4,500 request comparisons in 30 runs (10 seeds, rates 2, 6 and 10 req/s, 150 requests each), but it is an empirical property of this simulator, not a theorem: if it ever fails, check first for a cost term applied with the wrong sign.

## Chapter Summary

Serving splits into compute-bound prefill, which sets TTFT, and memory-bound decode, which sets TPOT. The decode floor is weight bytes over bandwidth, about 21 ms per token for a 70B BF16 model across two H100s at batch 1, before KV reads and communication. Continuous batching refills the batch every iteration, and shared weight reads make throughput rise with load until the batch is full. A fair measurement fixes and publishes the workload, reports TTFT, TPOT, percentiles, goodput under an SLO and cost, and sweeps the load. The simulator in this chapter reproduces those shapes and passes a hand-computed metric check and Little's law, but its numbers are illustrative.

## Self-Check Questions

1. Why is decode limited by memory bandwidth while prefill is not?
2. Why can a 70B BF16 model not decode faster than about 21 ms per token on two H100s at batch 1, under the stated dense BF16, HBM-streaming assumptions?
3. In the sweep, why does cost per million tokens fall as the rate rises, and why is the cheapest row not the best operating point?
4. Why must a throughput figure always come with a latency figure?
5. What does Little's law check in the simulator, and what does it not check?

## Worked Solutions

1. Prefill multiplies large matrices (many tokens against the weights), so it is arithmetic-heavy. Decode handles one token per sequence and must read all weights and the KV cache for little arithmetic, so time is set by bytes moved.
2. Under the stated streaming model, each step reads every weight once, so its modeled time is at least weight bytes divided by aggregate memory bandwidth: `140 GB / (2 x 3.35 TB/s) = 20.9 ms`. Real systems add KV reads, communication and launch overhead.
3. The GPU bill is fixed per hour, so more tokens per hour means a lower cost per token. At 16 requests per second the server is overloaded, so only 37 percent of requests meet the SLO; the useful operating point is the highest load that still meets it.
4. A server can raise throughput by letting requests queue into large batches, which makes users wait. Throughput alone hides that.
5. It checks that the simulator's time-average occupancy equals arrival rate times mean residence time, which catches bookkeeping errors. It does not check that the cost constants resemble a real GPU.

## Files and sources

- Code: `docs/examples/inference-systems/ch32_inference_harness.py` (`--test` for the self-tests, `--extras` for Experiments A and B)
- The decode floor uses the NVIDIA H100 SXM datasheet bandwidth (3.35 TB/s). The KV formula is the standard `2 x layers x kv_heads x head_dim x bytes`. The cost model constants are placeholders.
