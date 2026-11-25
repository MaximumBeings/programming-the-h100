# Chapter 36: Public Benchmark Teardown: vLLM, SGLang and TensorRT-LLM

[![Chapter 36: Benchmark the whole path](images/chapter-openers/36-inference.svg)](images/chapter-openers/36-inference.svg)


!!! abstract "Goal and prerequisites"

    Build a benchmark you could defend in public. You will get a load generator and statistics library that avoid the classic measurement errors, a test suite that proves they do, and a runbook for pointing the same client at vLLM, SGLang and TensorRT-LLM on real H100s. Read [Chapter 32](32-inference-engine-and-harness.md) (metrics and the simulated engine) first; this chapter replaces its simulator with a real network client.

!!! warning "Evidence level: read this first"

    - **Verified here (CPU):** the client, the streaming parsers, the percentile and bootstrap code, the Pareto and knee logic, the run manifest, and a mock three-engine study. 22 self-tests pass (14 pure, 8 against mock servers on localhost), plus 5 qualitative checks on the demonstrations.
    - **Invented:** the three "engines" in the mock study are made-up servers with made-up parameters. **No number in this chapter is a vLLM, SGLang or TensorRT-LLM result.**
    - **Timing output varies:** the mock servers run on wall-clock `asyncio` timing, so every latency number printed below is a *sample from one execution*. Another execution gives different numbers, and borderline verdicts can flip (Section 36.8 shows one that did). The tests therefore assert qualitative relations, never exact values.
    - **Untested on H100:** every launch command in Section 36.9 and the real teardown itself. Flags change between versions; confirm each against `--help` for the exact version you run. The original capture exercised mock servers. The [Chapter 47 lab](47-vllm-tuning-lab.md) and [dated record](validation-inference-2026-11-25.md) separately exercise its current streaming client against a small real vLLM model; the three-engine comparison remains untested.

## Worked system map

[![Benchmark the whole path: step-by-step hand trace](images/worked/ch36-inference.svg)](images/worked/ch36-inference.svg)


The November 25 corrections reject malformed/no-content streams, define one-token TPOT as zero, timestamp every terminal failure and include failed-request drain time in the measurement window. A timeout that outlasts the last successful request must not disappear from throughput, goodput or billed-duration accounting. Earlier raw captures remain unchanged.

## 36.1 Why most published benchmarks cannot be trusted

A comparison of serving engines fails in predictable ways. Every item below has produced a headline number that did not reproduce.

| # | Error | What it does to the result |
|---|---|---|
| 1 | Different precision or parallelism per engine (FP8 against BF16, TP=2 against TP=1) | Compares configurations, not engines |
| 2 | Different features on (prefix caching, speculative decoding, chunked prefill, CUDA graphs) | Credits an engine for a feature, or penalizes it for one |
| 3 | Closed-loop load ("N users, each waits for the last reply") | The client slows down when the server does, hiding queueing (coordinated omission, Section 36.5) |
| 4 | All requests sent at time zero | Measures a burst drain, not steady-state serving |
| 5 | Latency measured from when the client got around to sending | A stalled client erases its own delay |
| 6 | Failed or timed-out requests dropped from the statistics | Overstates throughput and goodput |
| 7 | One run, no warm-up exclusion, no spread | Reports noise as a difference |
| 8 | Averages instead of percentiles, or throughput without latency | Hides the tail users experience |
| 9 | Different clients, tokenizers or prompt counts per engine | Measures the benchmark tools |
| 10 | Output length decided by the model (different stopping per engine) | Different amounts of work per request |
| 11 | Unlabelled hardware state (clocks, driver, versions) | Not reproducible by anyone, including you next month |
| 12 | Only the winning workload shown | Selection bias |
| 13 | Prefix caching left on for one engine and off for another, with repeated prompts | Measures cache hits, not the engine |

The rest of the chapter is a defence against this list.

## 36.2 What each engine ships

All three engines expose an OpenAI-compatible HTTP server and ship their own benchmark client. That is both a convenience and a trap, because each client has its own defaults.

| Engine | Server | Vendor benchmark client | Notes from the documentation |
|---|---|---|---|
| vLLM | `vllm serve` | `vllm bench serve` | `--request-rate` with Poisson arrivals by default; `inf` sends everything at time zero; `--burstiness` switches to a gamma process; `--max-concurrency` caps in-flight requests; `--metadata` and `--save-detailed` record context and per-request data; goodput follows the DistServe paper's definition |
| SGLang | `python -m sglang.launch_server` | `python3 -m sglang.bench_serving` | Reports TTFT, TPOT and inter-token latency; its documentation says to start each server on the same hardware and precision and **run one at a time**; `bench_one_batch_server` sends a single batch and is explicitly not steady-state |
| TensorRT-LLM | `trtllm-serve` | `tensorrt_llm.serve.scripts.benchmark_serving` and `trtllm-bench` | Serving is configured with flags and a YAML file; the project's methodology is launch with `trtllm-serve`, then run the benchmark script |

**Design decision:** use **one** client for all three engines, written once, against the OpenAI-compatible endpoint. Use each vendor's client only as a cross-check, with the same model, prompt lengths, output lengths and request rate: on the same server the numbers should agree within noise. If they do not, find out why before publishing anything.

## 36.3 Experimental design

Decide these before running anything, and write them in the report.

| Variable | Rule | Why |
|---|---|---|
| Model and revision | Identical weights, identical checkpoint revision | |
| Precision | Same weight and KV-cache dtype, or report each engine's best supported option as a separate, labelled row | An FP8 engine against a BF16 engine is two experiments |
| Parallelism | Same GPU count and tensor-parallel degree | |
| Batch and scheduling limits | Equalize the maximum sequences and batched tokens where the engines expose them; otherwise report the values | Dominant drivers of the throughput and latency trade-off |
| Features | Prefix caching, speculative decoding, chunked prefill: all explicit, all reported. For a fair engine comparison on random prompts, disable prefix caching on every engine | Section 36.1 item 13 |
| Output length | Force a fixed output length per request (`ignore_eos`), drawn from a stated distribution | Equal work per request. The client marks a request **failed** if the output length differs, so an engine that ignores the flag shows up as failures, not as a silent speed-up |
| Prompts | Fixed seed, same prompt set, length counted with one tokenizer | |
| Sampling | Same temperature, top-p and seed settings | |
| Load | Open-loop Poisson arrivals at several rates, spanning idle to overloaded | Section 36.5 |
| Duration | At least 1,000 requests per point after warm-up for p99 claims (Section 36.6) | Percentiles need samples |
| Repeats | At least three, restarting the server each time | Run-to-run variation |
| Hardware state | Record GPU model, driver, CUDA version, clocks, power cap, persistence mode | Reproducibility |
| Isolation | Server and client in separate processes (ideally separate hosts or cores); one engine on the machine at a time | |

## 36.4 The client, piece by piece

The full file is `ch36_benchmark_teardown.py` ([download](examples/inference-systems/ch36_benchmark_teardown.py)). It uses only the Python standard library and speaks plain HTTP, so for an HTTPS endpoint put a local TLS-terminating proxy in front.

### Streaming parser

Servers stream tokens as Server-Sent Events, usually inside HTTP chunked encoding. The two layers can split an event anywhere, so the parser must not assume event boundaries line up with network reads, and it must cope with `\n`, `\r\n` and a `\r\n` pair cut between two reads. The parser accumulates bytes, normalizes line endings (holding back a trailing `\r`), and emits only events whose terminating blank line has arrived, joining multi-line `data:` fields with a newline as the specification says:

```python
def feed(self, chunk: bytes):
    data = self.buf + chunk
    hold = b""
    if data.endswith(b"\r"):          # may be the first half of CRLF: keep it for the next feed
        data, hold = data[:-1], b"\r"
    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ...                                # split on blank lines, collect data: fields
    self.buf = data + hold
```

Tests: four different streams (mixed LF and CRLF, **CRLF only**, multi-line data, and a stream ending in a partial event) are each cut at 200 random places, and every cutting must give the same events as the uncut stream. A trailing partial event is never emitted; the client then sees no `[DONE]` and counts the request as failed.

### Open-loop arrivals and intended time

Arrival times are drawn in advance from a Poisson process (exponential gaps). Each request records when the schedule said to send it (`intended`) and when it was actually sent (`sent`). Latency is measured from `intended`, so if the client itself stalls (garbage collection, a busy event loop, a slow DNS lookup) the delay is charged to the request rather than hidden.

### The request, and what counts as success

The payload is a real OpenAI-style streaming completion request: model name, a prompt of the requested length, `max_tokens`, temperature 0, `stream: true`, `stream_options: {"include_usage": true}` and, by default, `ignore_eos: true` (accepted by vLLM and SGLang; confirm for your engine, or pass `--no-ignore-eos`).

A request is successful only if the stream ended with `[DONE]`, at least one content event arrived, no parse error occurred, and the server-reported token count equals the requested output length. The token count requires the server's `usage.completion_tokens`, because real engines can merge several tokens into one event or send empty deltas. Content events are retained separately as `nevents`; without usage the request fails instead of treating events as tokens. HTTP errors, timeouts, truncated streams and wrong lengths are all failures, recorded with a reason. Failures stay in the denominator of every rate and attainment figure.

Timestamps: `first` is the arrival time of the first content event, `end` the terminal response/connection timestamp (or failure timestamp), `itl` the gaps between content events. TTFT is `first - intended`, TPOT is `(end - first) / (ntok - 1)` when more than one token is emitted, and zero for a one-token response, end-to-end latency is `end - intended`. If an engine merges tokens into events, ITL is per event, not per token; TPOT is unaffected.

### Running it against a real server

```bash
python3 /full/path/ch36_benchmark_teardown.py run \
    --url http://127.0.0.1:8000/v1/completions --model <served-model-name> \
    --rate 10 --n 1200 --warmup 100 --seed 1 --n-in 512 --n-out 128 \
    --slo-ttft 0.5 --slo-tpot 0.05 --usd-per-gpu-hour <your-price> --gpus 2 \
    --out /full/path/results/engine-a_rate10_run1.jsonl
```

It prints the summary, a count of failure reasons, a run manifest, and writes every per-request record to the JSONL file so any number can be recomputed. Prompts are built from the word "hello" repeated, so token counts are approximate on the input side; read the true prompt length from the server's usage field if you need it. (Unique random prompts are better when measuring prefix caching; extend `make_payload` for that.)

## 36.5 Coordinated omission, demonstrated

A closed-loop client with N workers sends a new request only when a reply comes back, so when the server slows down the client sends less. The queue never builds as it would for independent users, and measured latency stays modest. An open-loop client keeps the arrival schedule, so the queue builds as it would in production.

The mock server has 4 slots and takes about 0.43 s per request, so its capacity is **about 9 requests per second**. The demonstration therefore runs in three regimes.

### Worked Example 36.5.1: below and above capacity

```
  open-loop, Poisson arrivals, capacity of the mock server about 9 req/s:
    below capacity (5 req/s)  : ttft_p99 =    415.2 ms   e2e_p99 =   1314.6 ms   (n=87)
    2x overload, short run    : ttft_p99 =   2349.0 ms   e2e_p99 =   2645.3 ms   (n=37)
    2x overload, long run     : ttft_p99 =  10059.7 ms   e2e_p99 =  10359.3 ms   (n=157)
  closed-loop, 4 workers      : ttft_p99 =     42.4 ms   e2e_p99 =   1074.2 ms   (n=87)
```

(Sample output; yours will differ.) Read it in three steps:

1. **Below capacity** the open-loop p99 TTFT is 415 ms, because Poisson arrivals still cluster and a burst waits for a slot. The closed-loop client shows 42 ms. The gap (roughly tenfold here) is real, and it is the honest price of random arrivals; it is also bounded, unlike the overload case below.
2. **Above capacity there is no steady state.** Doubling the run length from 37 to 157 requests quadruples the p99 TTFT (2.3 s to 10 s). The right statement is not "p99 is 10 s at 18 req/s" but "the system is unstable at 18 req/s: latency grows without bound for as long as the load lasts". A benchmark point beyond the knee is a statement about how long you ran, not about the engine.
3. **The closed-loop client reports 42 ms for the same server**, because it never offers more than 4 requests at a time. It answers a different question (how fast is it for 4 users who always wait?) and cannot reveal overload.

### Worked Example 36.5.2: a stalled client

The client blocks its own event loop for 0.6 s while 100 requests are being issued at 40 per second. About 24 requests should have been sent during the stall; they go out late, immediately afterwards.

```
  0.6 s client stall at request 40 (of 100, 40 req/s offered):
    share with TTFT > 300 ms measured from SEND time     =  3.2%
    share with TTFT > 300 ms measured from INTENDED time = 12.6%
    mean TTFT: from send   36.7 ms   from intended   92.6 ms
```

Measured from the send time, the late requests look healthy and the client's own stall vanishes from the data. Measured from the intended time, the stall is counted, as a user who clicked at the scheduled moment would have experienced it. This is also why a benchmark client must be lightweight and isolated: a stalled client is charged to the server under test.

The exact percentages depend on timing; the self-checks assert only that the open-loop p99 below capacity stays within 30 times the closed-loop figure, that the intended-time share and mean exceed the send-time ones, that the overloaded open-loop p99 is at least 5 times the closed-loop figure, and that it grows with run length.

A failed request receives a terminal timestamp even on connection error or timeout. The observation window ends at the last terminal timestamp across **all** requests, while useful token throughput counts only successful replies. Failed-request tail time still consumes elapsed and billed time. This prevents a long timeout from making the same successful work appear faster or cheaper.

## 36.6 Statistics

### Warm-up, per run

The first requests hit cold caches, compile kernels or capture CUDA graphs. Discard them **by arrival order**, a fixed count chosen in advance, **from each run separately**, and report the number discarded. An earlier version of this harness pooled the runs first and then dropped the earliest requests overall, which removed all warm-up from the first run and none from the second. A self-test now builds two runs 1,000 s apart and checks that 10 are dropped from each. Throughput is computed over the **sum of the per-run windows**, so the idle gap between runs does not dilute tokens per second or inflate cost per token; a second self-test uses the same two runs to check that.

### Percentiles

Use linear interpolation between order statistics (the default of common numerical libraries). The self-test compares the implementation with NumPy for p50 and p99. Report p50, p99 and, where the sample allows, p99.9, and report the sample size next to every percentile, because the p99 of 50 requests is a single observation.

### Bootstrap confidence intervals, and where they fail

Resample the per-request values with replacement many times, compute the statistic each time, and take the 2.5th and 97.5th percentiles. The self-test measures how often the 95 percent interval contains the true value, on lognormal data with a known answer:

```
bootstrap coverage of true value (nominal 0.95): median n=120 -> 0.92; p99 n=160 -> 0.77; p99 n=1000 -> 0.85
```

- **Median:** close to nominal at 120 samples.
- **p99:** poor. At 160 requests the "95 percent" interval contains the truth only about 77 percent of the time, and at 1,000 requests about 85 percent. A resample can never exceed the sample maximum, so the interval is too narrow exactly where the tail is thin.

The consequence is a rule: **treat the printed p99 interval as an uncalibrated resampling diagnostic, not a guaranteed 95% interval or a mathematical bound on the uncertainty.** Larger samples and independent repeated runs help reveal instability; 1,000 requests and three repetitions are practical starting points, not a coverage guarantee. Look at the per-run p99 values printed beside the interval. In the first version of the mock study, engine C at 15 req/s had a run whose p99 (70 ms) lay outside its own interval [87, 200]: that was this effect, visible in the table.

### Comparing two engines

The mock demonstration bootstraps the **difference** of the statistic using individual requests. An interval excluding zero separates the samples under that request-IID resampling procedure; it does not by itself establish a reliable population difference between serving systems. Requests sharing queues, caches and batches are correlated. Use independent workload runs as the replication unit, preserve pairing when runs use matched workloads, and consider a justified block/bootstrap method for within-run dependence. With only three runs, report descriptive per-run differences and ranges rather than a precise confidence claim. The printed mock table’s “95% CI” labels identify the IID diagnostic, not validated coverage for the serving workload.

## 36.7 Pareto frontiers, knees and cost

One number cannot rank engines: they trade latency, throughput and cost, and the trade changes with load.

- **Pareto frontier:** a point is dominated if another is no worse on every axis and better on at least one. The implementation is tested on 200 random points: every kept point is non-dominated, and every dropped point is dominated.
- **Feasibility first:** only points that meet the SLO belong on the frontier. A cheap point at which half the requests miss their latency target is a failure, not a competitor.
- **Knee:** the highest offered rate at which SLO attainment stays at or above a threshold (here 99 percent, with failures counted as misses). Capacity planning uses the knee, not peak throughput.
- **Cost:** `usd_per_gpu_hour x GPUs x wall_hours / million_output_tokens`, over the sum of run windows. State the price and where it comes from. Below the knee, cost per token mostly reflects idle capacity (the same GPUs serve fewer tokens), so it falls as load rises until the knee; comparing cost at unequal utilization compares loads, not engines.

## 36.8 A mock teardown, end to end

Three **invented** engines (different per-token latency, concurrency capacity and GPU count) are driven at 15, 35 and 70 requests per second, three repeats of 150 requests each (450 per point, 420 after warm-up). The table is a **sample from one execution**.

```
  engine     rate     n  ttft_p99(ms)  [95% CI]            tpot_p99(ms)  tok/s  fail  attain  $/Mtok  per-run p99 (ms)
  engine-A    15   420         40.2  [  36.3,   45.6]          6.1    681     0    1.00    3.26  38 / 41 / 38
  engine-A    35   420         83.0  [  65.2,  120.8]          7.1   1571     0    1.00    1.41  56 / 79 / 117
  engine-A    70   420       1257.5  [1216.7, 1312.2]          7.3   1979     0    0.45    1.12  1071 / 1226 / 1294
  engine-B    15   420         40.3  [  36.1,   44.4]          7.2    686     0    1.00    3.24  39 / 41 / 38
  engine-B    35   420         38.9  [  36.1,   45.1]         12.5   1555     0    1.00    1.43  36 / 43 / 37
  engine-B    70   420       1242.9  [1165.0, 1301.1]         15.0   1871     0    0.52    1.19  1074 / 1170 / 1300
  engine-C    15   420        151.0  [ 119.9,  183.9]          6.5    676     0    1.00    1.64  142 / 157 / 117
  engine-C    35   420       2206.9  [2155.4, 2281.7]          6.6   1111     0    0.21    1.00  1671 / 2065 / 2272
  engine-C    70   420       4120.4  [4006.7, 4210.4]          6.6   1104     0    0.10    1.01  4056 / 3979 / 4209

  knee: engine-A 35 req/s, engine-B 35 req/s, engine-C 15 req/s
  p99 TTFT difference A-B at 35 req/s: 95% CI [25.2, 82.1] ms -> engines differ beyond resampling noise
  SLO-feasible Pareto-optimal (cost, p99 TTFT, throughput): engine-A@35, engine-B@35
  excluded as infeasible (SLO attainment < 99%): engine-A@70, engine-B@70, engine-C@35, engine-C@70
  note: the lowest $/Mtok anywhere is engine-C@35 (1.00); it is NOT SLO-feasible
```

How to read it, and what each line teaches:

1. **The A-versus-B comparison is fragile.** Here the interval excludes zero, but look at the per-run p99 for engine A at 35 req/s: 56, 79 and 117 ms. One run in three is twice another. An earlier version of this study used 90 requests per run and two runs; its interval for the same comparison included zero in one execution and excluded it in another. The honest summary is "B is probably better at 35 req/s in the tail, by tens of milliseconds, and the claim needs more repeats and requests". Do not pick the favourable execution.
2. **The cheapest point is not a valid choice.** Engine C at 35 req/s has the lowest cost per million tokens in the table (1.00), yet only 21 percent of its requests meet the SLO. The program prints this explicitly: the cheapest point anywhere is not SLO-feasible. Engine A at 70 req/s (1.12) is also cheaper than anything feasible and also infeasible (45 percent attainment).
3. **Different knees.** A and B sustain 35 req/s within SLO; C only 15. Engine C's saturation also shows in its throughput: it stops at about 1,100 tokens per second whatever the offered load.
4. **Failures are zero here** because the mock servers inject none. The socket self-tests show the other path: with 30 percent injected 503 errors, 15 of 60 requests fail and attainment drops to 0.75.
5. **Tail intervals are optimistic.** Section 36.6 measured p99 interval coverage below 90 percent even at 1,000 requests; the intervals above come from 420 requests.

The mock engines were built to produce these shapes. Real engines will produce different ones, and some of the differences will be configuration, not engine quality.

## 36.9 Runbook for real H100s (untested)

!!! warning "UNTESTED ON H100"

    None of this was executed. Flags below follow each project's documentation at the time of writing but change between releases; run `--help` for your version and record the version in the report. The client in this chapter has only been run against the mock server.

### 1. Capture and fix the environment

```bash
mkdir -p /full/path/env
nvidia-smi -q > /full/path/env/nvidia-smi-full.txt
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,persistence_mode,power.limit --format=csv > /full/path/env/gpu.csv
sudo nvidia-smi -pm 1                       # persistence mode, so the driver stays loaded between runs
sudo nvidia-smi -lgc <min_mhz>,<max_mhz>    # optional: lock SM clocks; choose values for your card, or just record them
pip freeze > /full/path/env/pip-freeze.txt  # inside each engine's container or venv
```

Pin each engine by container image tag or exact package version. Record whether other processes share the GPU. During runs, sample `nvidia-smi --query-gpu=clocks_throttle_reasons.active,temperature.gpu,power.draw --format=csv -l 1` (the field name is `clocks_event_reasons.active` on newer drivers) so thermal or power throttling shows up in the data.

### 2. Launch one engine at a time, same model, same GPUs, prefix caching off

```bash
# vLLM  (check: vllm serve --help)
vllm serve <model> --tensor-parallel-size 2 --port 8000 --no-enable-prefix-caching \
    --max-num-seqs <N> --max-num-batched-tokens <N> --max-model-len <N>

# SGLang  (check: python -m sglang.launch_server --help)
python -m sglang.launch_server --model-path <model> --tp-size 2 --port 30000 --disable-radix-cache \
    --max-running-requests <N> --max-total-tokens <N>

# TensorRT-LLM  (check: trtllm-serve --help). Engine settings go in a YAML file; the flag that passes it
# is --config in recent releases and --extra_llm_api_options in older ones.
trtllm-serve <model> --tp_size 2 --max_batch_size <N> --max_num_tokens <N> --max_seq_len <N> --config engine.yml
```

Equalize the scheduling limits where each engine exposes them (the placeholders above) and write the values in the report. Record any KV-cache dtype and quantization option explicitly. If an engine cannot disable prefix caching, use unique random prompts per request and say so.

### 3. Point the one client at each server

Run the Section 36.4 command against each server with the same seed, prompt and output lengths, and rates. Restart the server between repeats. Name the JSONL files by engine, rate and run so a script can aggregate them.

### 4. Cross-check with the vendor clients on the same server

```bash
vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model <model> \
    --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
    --num-prompts 1200 --request-rate 10 --seed 1 --metadata engine=vllm
python3 -m sglang.bench_serving --backend sglang --port 30000 --model <model> \
    --dataset-name random --random-input-len 512 --random-output-len 128 \
    --num-prompts 1200 --request-rate 10
```

Match model, input and output lengths, request rate and seed to the chapter's client, and confirm the exact flag names with `--help` (the random-dataset and ignore-eos options differ between versions). Do not use `--request-rate inf` for a steady-state comparison: in vLLM's client that sends every request at time zero, which measures draining a burst. Use `--max-concurrency` only if you intend a capped, closed-loop-style test, and label it. If the vendor client and this one disagree by more than the run-to-run spread on the same server, resolve that first.

### 5. Sweep and decide

Run at least five offered rates from well below to well above the expected knee, three or more repeats each, then produce the Section 36.8 table and the report below.

## 36.10 Report template

| Section | Contents |
|---|---|
| Question | One sentence: "Which engine gives the lowest cost per million tokens at p99 TTFT under 500 ms and p99 TPOT under 50 ms for chat-length requests on 2 x H100?" |
| Environment | GPU model and count, driver, CUDA, host CPU, container tags, clocks, power cap, throttle log |
| Engines and configs | Exact launch commands, scheduling limits, precision, features on and off |
| Workload | Arrival process and rates, prompt and output length distributions with seed, number of requests per point and warm-up discarded per run |
| Results | Per engine and rate: TTFT and TPOT p50 and p99 with intervals and sample sizes, throughput, failures by reason, SLO attainment, cost per million tokens, per-run spread |
| Frontier | SLO-feasible Pareto set and the knee per engine |
| Where each engine loses | At least one workload or rate where each engine is worst, and why |
| Threats to validity | Items from Section 36.1 you could not control |
| Reproduction | Repository with scripts, raw per-request JSONL, manifest and a single command |

## 36.11 Failure-mode catalogue

| Symptom | Likely cause | Check |
|---|---|---|
| Throughput far below the vendor's published figure | Different workload (shorter outputs, higher concurrency), batch limits, or a closed-loop client | Run the vendor's client on the same server with matched flags |
| p99 TTFT swings between runs | Too few requests, noisy host, thermal or power throttling | Increase requests, record clocks and throttle reasons, repeat |
| Many requests fail with "wrong-length stream" | Engine ignores `ignore_eos`, or the usage field is absent so merged events are miscounted | Inspect the JSONL `err` and `nevents` fields; try `--no-ignore-eos` and compare; check that the server returns usage in streaming mode |
| Zero failures at every load | Failures swallowed | Inject errors (mock) or overload deliberately and confirm they are counted |
| Latency improves at high load | Closed-loop client | Switch to open-loop |
| Latency grows with run length | The point is above the knee | Report it as unstable, not as a p99 |
| Events never parse | Server sends a different framing (for example CRLF) or non-JSON events | The parser handles CR and LF; check `err == "unparseable event"` and look at the raw bytes |
| Engines differ only in the tail | Scheduling policy differences | Compare TPOT distributions and queueing |
| One engine uses much less memory and "wins" | Smaller KV cache, so a lower concurrency ceiling | Compare maximum concurrent sequences |
| Results change after a driver update | Environment drift | Re-run, and keep the manifest with the data |
| Suspiciously good latency on repeated prompts | Prefix cache hits | Disable prefix caching or use unique prompts |

<!-- BEGIN COMPLETE INFERENCE SOURCE -->
## Complete runnable code

The complete implementation is shown below, including its checks and entry point. The listing and downloadable `.py` file contain exactly the same source. Copy the file under the displayed filename; keep supporting files in the same directory.

[Download all inference examples and supporting modules](examples/inference-systems/inference-systems.zip) or read the [environment and command guide](examples/inference-systems/README.md).

The [follow-up agent audit](inference-adversarial-review-2026-11-25.md) records the corrections, fresh CPU captures and hardware limits for this source.

Read it in three passes: start with `Rec` and `Target` to identify the state; trace `one_request` and `run_open_loop` for one request or token; then follow `summarize` and `selftests` to see how the result is checked.

From the repository root:

```bash
python3 docs/examples/inference-systems/ch36_benchmark_teardown.py --selftest
# Run the full chapter demonstrations:
python3 docs/examples/inference-systems/ch36_benchmark_teardown.py
```

### ch36_benchmark_teardown.py

[Download ch36_benchmark_teardown.py](examples/inference-systems/ch36_benchmark_teardown.py)

```python title="ch36_benchmark_teardown.py" linenums="1"
"""Chapter 36: a rigorous serving-benchmark harness, tested against mock OpenAI-compatible streaming servers.

Run:   python3 ch36_benchmark_teardown.py            # self-tests + demos + mock three-engine teardown (a few minutes, CPU)
       python3 ch36_benchmark_teardown.py --selftest # self-tests only (about 20 s)
       python3 ch36_benchmark_teardown.py run --url http://127.0.0.1:8000/v1/completions --model NAME \
               --rate 10 --n 600 --warmup 50 --seed 1 --out run.jsonl     # drive a REAL OpenAI-compatible server

What is real here: the client (open-loop Poisson load, intended-time latency, streaming timestamps, failure
accounting), the statistics (percentiles, bootstrap CIs), the Pareto/knee logic, the SSE and chunked parsers, and
the run manifest. What is NOT real: the three "engines" are mock servers with invented parameters. No vLLM, SGLang
or TensorRT-LLM result is produced or implied by this file.
"""
import argparse
import asyncio
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit
from dataclasses import dataclass, field, asdict


# =============================================================== 1. streaming parsers
class SSEParser:
    """Incremental Server-Sent-Events parser. Feed arbitrary byte chunks; get complete event payloads back.

    Handles LF, CRLF and lone-CR line endings (including a CR/LF pair split across two chunks), ignores comment
    lines, and joins multi-line 'data:' fields with '\n' as the SSE spec says. An event is complete only when its
    terminating blank line has arrived; a trailing partial event is never emitted (the client then sees no [DONE]
    and counts the request as failed)."""
    def __init__(self):
        self.buf = b""

    def feed(self, chunk: bytes):
        data = self.buf + chunk
        hold = b""
        if data.endswith(b"\r"):                       # may be the first half of CRLF: keep it for the next feed
            data, hold = data[:-1], b"\r"
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        out = []
        while True:
            i = data.find(b"\n\n")
            if i < 0:
                break
            event, data = data[:i], data[i + 2:]
            fields = []
            for line in event.split(b"\n"):
                if line.startswith(b"data:"):
                    v = line[5:]
                    fields.append((v[1:] if v.startswith(b" ") else v).decode())
            if fields:
                out.append("\n".join(fields))
        self.buf = data + hold
        return out


async def read_body_chunks(reader, headers):
    """Yield body bytes as they arrive, handling both 'Transfer-Encoding: chunked' and close-delimited bodies."""
    if headers.get("transfer-encoding", "").lower() == "chunked":
        while True:
            size_line = await reader.readline()
            if not size_line:
                return
            size = int(size_line.strip().split(b";")[0], 16)
            if size == 0:
                await reader.readline()
                return
            data = await reader.readexactly(size)
            await reader.readexactly(2)                      # trailing CRLF
            yield data
    else:
        while True:
            data = await reader.read(4096)
            if not data:
                return
            yield data


# =============================================================== 2. mock engine (stands in for a real server)
@dataclass
class EngineProfile:
    name: str
    base_ms: float            # per-token base latency
    per_seq_ms: float         # extra per-token latency per concurrently running sequence
    capacity: int             # max concurrently running sequences (others queue)
    prefill_ms_per_100tok: float = 4.0
    error_rate: float = 0.0
    gpus: int = 1
    chunked: bool = True
    merge: int = 1             # tokens per SSE event (real engines sometimes batch several)
    truncate_prob: float = 0.0 # probability the stream is cut short without [DONE]


class MockEngine:
    def __init__(self, prof: EngineProfile, seed=0):
        self.p, self.rng = prof, random.Random(seed)
        self.sem = asyncio.Semaphore(prof.capacity)
        self.active = 0
        self.server = None
        self.port = None

    async def start(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self):
        self.server.close()
        await self.server.wait_closed()

    @staticmethod
    def _chunk(data: bytes) -> bytes:
        return f"{len(data):x}\r\n".encode() + data + b"\r\n"

    async def handle(self, reader, writer):
        try:
            await reader.readline()                                   # request line
            clen = 0
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b""):
                    break
                if line.lower().startswith(b"content-length:"):
                    clen = int(line.split(b":")[1])
            body = json.loads(await reader.readexactly(clen))
            n_out = int(body["max_tokens"])
            n_in = len(body["prompt"].split()) if "prompt" in body else int(body.get("prompt_tokens", 128))
            if self.rng.random() < self.p.error_rate:
                writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return
            hdr = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            hdr += b"Transfer-Encoding: chunked\r\n\r\n" if self.p.chunked else b"Connection: close\r\n\r\n"
            writer.write(hdr)
            put = self._chunk if self.p.chunked else (lambda b: b)
            cut = self.rng.random() < self.p.truncate_prob
            n_send = n_out // 2 if cut else n_out
            async with self.sem:                                      # queueing happens here
                self.active += 1
                try:
                    await asyncio.sleep(n_in / 100 * self.p.prefill_ms_per_100tok / 1e3)
                    sent = 0
                    while sent < n_send:
                        k = min(self.p.merge, n_send - sent)
                        await asyncio.sleep(k * (self.p.base_ms + self.p.per_seq_ms * self.active) / 1e3)
                        writer.write(put(b'data: {"choices":[{"text":"' + b"x" * k + b'"}]}\n\n'))
                        await writer.drain()
                        sent += k
                finally:
                    self.active -= 1
            if cut:
                return                                                # connection closed mid-stream, no [DONE]
            writer.write(put(b'data: {"choices":[],"usage":{"completion_tokens":%d}}\n\ndata: [DONE]\n\n' % n_out))
            if self.p.chunked:
                writer.write(b"0\r\n\r\n")
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()


# =============================================================== 3. client: open-loop and closed-loop load
@dataclass
class Rec:
    intended: float                       # when the schedule said to send
    sent: float = math.nan                # when we actually sent
    first: float = math.nan               # first streamed token event
    end: float = math.nan
    ntok: int = 0                         # output tokens: server-reported usage if present, else content events
    nevents: int = 0                      # content events seen (may be < ntok if the server merges tokens)
    ok: bool = False
    err: str = ""
    itl: list = field(default_factory=list)


@dataclass
class Target:
    host: str = "127.0.0.1"
    port: int = 8000
    path: str = "/v1/completions"
    model: str = "mock"
    ignore_eos: bool = True               # force exact output length so every engine does equal work

    @staticmethod
    def from_url(url, model="mock", ignore_eos=True):
        u = urlsplit(url)
        if u.scheme != "http":
            raise ValueError("this client speaks plain http; put a local TLS-terminating proxy in front of https servers")
        return Target(u.hostname, u.port or 80, u.path or "/v1/completions", model, ignore_eos)


def make_payload(t: Target, n_in, n_out):
    p = {"model": t.model, "prompt": " ".join(["hello"] * n_in), "max_tokens": n_out, "temperature": 0,
         "stream": True, "stream_options": {"include_usage": True}}
    if t.ignore_eos:
        p["ignore_eos"] = True                                        # accepted by vLLM and SGLang; check your engine
    return json.dumps(p).encode()


async def one_request(target, n_in, n_out, rec: Rec, timeout=30.0, payload=None):
    if isinstance(target, int):                                       # backwards compatible: bare port
        target = Target(port=target)
    writer = None
    try:
        async def go():
            nonlocal writer
            reader, writer = await asyncio.open_connection(target.host, target.port)
            body = payload or make_payload(target, n_in, n_out)
            rec.sent = time.perf_counter()
            writer.write(f"POST {target.path} HTTP/1.1\r\nHost: {target.host}:{target.port}\r\n"
                         f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
            await writer.drain()
            status = (await reader.readline()).decode()
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b""):
                    break
                k, _, v = line.decode().partition(":")
                headers[k.strip().lower()] = v.strip()
            if " 200 " not in status:
                rec.err = status.strip() or "empty status line"
                writer.close()
                return
            parser, last, done, usage = SSEParser(), None, False, None
            async for chunk in read_body_chunks(reader, headers):
                for ev in parser.feed(chunk):
                    now = time.perf_counter()
                    if ev.strip() == "[DONE]":
                        done = True
                        continue
                    try:
                        obj = json.loads(ev)
                    except ValueError:
                        rec.err = "unparseable event"
                        continue
                    if isinstance(obj, dict) and obj.get("usage"):
                        usage = obj["usage"].get("completion_tokens", usage)
                    text = ""
                    for ch in (obj.get("choices") or []) if isinstance(obj, dict) else []:
                        text += ch.get("text") or (ch.get("delta") or {}).get("content") or ""
                    if text:                                          # one content-bearing event
                        if math.isnan(rec.first):
                            rec.first = now
                        elif last is not None:
                            rec.itl.append(now - last)
                        last = now
                        rec.nevents += 1
            rec.end = time.perf_counter()
            rec.ntok = usage if usage is not None else 0             # SSE events are not token counts
            rec.ok = (done and not rec.err and usage is not None and rec.ntok == n_out and
                      math.isfinite(rec.first) and rec.end >= rec.first)                       # truncated stream or wrong length = FAILURE
            if not rec.ok:
                rec.err = f"incomplete or wrong-length stream: {rec.ntok}/{n_out} tokens, done={done}"
            writer.close()
        await asyncio.wait_for(go(), timeout)
    except asyncio.TimeoutError:
        rec.err = "timeout"
    except Exception as e:                                           # noqa: BLE001 - benchmark must not crash
        rec.err = f"{type(e).__name__}: {e}"
    finally:
        # Failures consume elapsed/billed time too, including connection failures
        # and timeouts. Always retain a terminal timestamp and close the socket.
        if not math.isfinite(rec.end):
            rec.end = time.perf_counter()
        if writer is not None:
            writer.close()
    return rec


async def run_open_loop(port, rate, n_req, seed, n_in_med=256, n_out_med=48, stall_at=None, stall_s=0.0):
    """Poisson arrivals scheduled in advance. Each request is launched at its INTENDED time regardless of how
    earlier requests are doing. stall_at/stall_s inject a client-side stall (to demonstrate coordinated omission)."""
    rng = random.Random(seed)
    t, sched = 0.0, []
    for _ in range(n_req):
        t += rng.expovariate(rate)
        sched.append((t, max(8, int(rng.lognormvariate(math.log(n_in_med), .5))),
                      max(4, int(rng.lognormvariate(math.log(n_out_med), .4)))))
    t0 = time.perf_counter() + 0.05
    recs, tasks = [], []
    for i, (at, ni, no) in enumerate(sched):
        rec = Rec(intended=t0 + at)
        recs.append(rec)
        delay = rec.intended - time.perf_counter()
        if stall_at is not None and i == stall_at:
            time.sleep(stall_s)                                      # blocks the whole client event loop
            delay = rec.intended - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one_request(port, ni, no, rec)))
    await asyncio.gather(*tasks)
    return recs


async def run_closed_loop(port, concurrency, n_req, seed, n_in_med=256, n_out_med=48):
    """N workers; each sends its next request only after the previous one finishes. (Self-throttling.)"""
    rng = random.Random(seed)
    jobs = [(max(8, int(rng.lognormvariate(math.log(n_in_med), .5))),
             max(4, int(rng.lognormvariate(math.log(n_out_med), .4)))) for _ in range(n_req)]
    recs, it = [], iter(jobs)

    async def worker():
        for ni, no in it:
            r = Rec(intended=time.perf_counter())                    # "intended" == now: that is the omission
            recs.append(r)
            await one_request(port, ni, no, r)
    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return recs


# =============================================================== 4. metrics and statistics
def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return math.nan
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _collect(recs, warmup, slo_ttft, slo_tpot, from_intended):
    """Per-RUN statistics. Warm-up is discarded per run, by arrival order, so every run loses its own cold start."""
    recs = sorted(recs, key=lambda r: r.intended)[warmup:]
    ok = [r for r in recs if r.ok]
    base = (lambda r: r.intended) if from_intended else (lambda r: r.sent)
    ttft = [r.first - base(r) for r in ok]
    tpot = [(r.end - r.first) / (r.ntok - 1) if r.ntok > 1 else 0.0 for r in ok]
    return dict(n=len(recs), ok=len(ok), ttft=ttft, tpot=tpot, e2e=[r.end - base(r) for r in ok],
                wall=(max(r.end for r in recs) - min(r.intended for r in recs)) if recs else 0.0,
                toks=sum(r.ntok for r in ok),
                good=sum(1 for a, b in zip(ttft, tpot) if a <= slo_ttft and b <= slo_tpot))


def _finalize(parts, usd_per_gpu_hour, gpus):
    n = sum(p["n"] for p in parts); nok = sum(p["ok"] for p in parts)
    ttft = [x for p in parts for x in p["ttft"]]; tpot = [x for p in parts for x in p["tpot"]]
    e2e = [x for p in parts for x in p["e2e"]]
    wall = sum(p["wall"] for p in parts)                      # sum of per-run windows: the gap BETWEEN runs is excluded
    toks, good = sum(p["toks"] for p in parts), sum(p["good"] for p in parts)
    out = dict(n=n, failures=n - nok, success_rate=nok / n if n else math.nan,
               ttft_p50=percentile(ttft, .5), ttft_p99=percentile(ttft, .99),
               tpot_p50=percentile(tpot, .5), tpot_p99=percentile(tpot, .99), e2e_p99=percentile(e2e, .99),
               tok_s=toks / wall if wall else 0.0, goodput_rps=good / wall if wall else 0.0,
               slo_attain=good / n if n else math.nan,          # denominator = ALL requests, failures included
               ttft_samples=ttft)
    if usd_per_gpu_hour and toks:
        out["usd_per_mtok"] = usd_per_gpu_hour * gpus * wall / 3600 / (toks / 1e6)
    return out


def summarize(recs, warmup=10, slo_ttft=0.5, slo_tpot=0.05, from_intended=True, usd_per_gpu_hour=None, gpus=1):
    return _finalize([_collect(recs, warmup, slo_ttft, slo_tpot, from_intended)], usd_per_gpu_hour, gpus)


def summarize_runs(runs, warmup=10, slo_ttft=0.5, slo_tpot=0.05, from_intended=True, usd_per_gpu_hour=None, gpus=1):
    """Pool several repeats of the same configuration: warm-up per run, throughput over the sum of run windows."""
    return _finalize([_collect(r, warmup, slo_ttft, slo_tpot, from_intended) for r in runs], usd_per_gpu_hour, gpus)


def bootstrap_ci(xs, stat, B=800, alpha=0.05, seed=0):
    """Percentile bootstrap over requests. Good for medians and means; for p99 it under-covers unless n is large
    (about 1000 or more), because a resample can never exceed the sample maximum. See the coverage self-test."""
    rng = random.Random(seed)
    n = len(xs)
    vals = sorted(stat(rng.choices(xs, k=n)) for _ in range(B))
    return vals[int(B * alpha / 2)], vals[int(B * (1 - alpha / 2)) - 1]


def diff_ci(a, b, stat, B=800, alpha=0.05, seed=0):
    """CI for stat(a) - stat(b). If it excludes 0, the engines differ beyond resampling noise (for THIS workload)."""
    rng = random.Random(seed)
    vals = sorted(stat(rng.choices(a, k=len(a))) - stat(rng.choices(b, k=len(b))) for _ in range(B))
    return vals[int(B * alpha / 2)], vals[int(B * (1 - alpha / 2)) - 1]


def pareto(points, minimize, maximize=()):
    """Indices of non-dominated points. A point is dominated if another is no worse on every axis and better on one."""
    def key(p):
        return [p[m] for m in minimize] + [-p[m] for m in maximize]
    keys = [key(p) for p in points]
    keep = []
    for i, ki in enumerate(keys):
        dominated = any(all(kj[t] <= ki[t] for t in range(len(ki))) and any(kj[t] < ki[t] for t in range(len(ki)))
                        for j, kj in enumerate(keys) if j != i)
        if not dominated:
            keep.append(i)
    return keep


def knee(rows, min_attain=0.99):
    """Highest offered rate whose SLO attainment (failures counted) stays at or above min_attain."""
    good = [r for r in rows if r["slo_attain"] >= min_attain]
    return max((r["rate"] for r in good), default=None)


# =============================================================== 5. run manifest (reproducibility)
def manifest(extra=None):
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
        except Exception:                                            # noqa: BLE001
            return None
    smi = None
    if shutil.which("nvidia-smi"):
        # field names differ between drivers; try the richer query first
        for q in ("name,driver_version,clocks.sm,clocks.mem,persistence_mode,power.limit,clocks_throttle_reasons.active",
                  "name,driver_version,clocks.sm,clocks.mem,persistence_mode,power.limit"):
            smi = sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"])
            if smi:
                break
    m = dict(python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine(),
             argv=sys.argv, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), nvidia_smi=smi)
    m.update(extra or {})
    m["config_hash"] = hashlib.sha256(json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return m


# =============================================================== 6. self-tests
def selftests():
    results = []

    def check(name, cond):
        results.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = random.Random(1)
    # SSE parser: result independent of how the byte stream is cut, for LF, CRLF-only and multi-line data
    streams = {
        "LF/CRLF mix": (b'data: {"a":1}\n\ndata: {"b":2}\r\n\r\n: comment\n\ndata: [DONE]\n\n',
                        ['{"a":1}', '{"b":2}', "[DONE]"]),
        "CRLF only": (b'data: {"a":1}\r\n\r\ndata: {"b":2}\r\n\r\ndata: [DONE]\r\n\r\n',
                      ['{"a":1}', '{"b":2}', "[DONE]"]),
        "multi-line data": (b'data: line1\ndata: line2\n\ndata: [DONE]\n\n', ["line1\nline2", "[DONE]"]),
        "partial last event": (b'data: one\n\ndata: tw', ["one"]),
    }
    for name, (stream, expect) in streams.items():
        whole = SSEParser().feed(stream)
        same = True
        for _ in range(200):
            p, out, i = SSEParser(), [], 0
            while i < len(stream):
                j = i + rng.randint(1, 7)
                out += p.feed(stream[i:j]); i = j
            same &= (out == whole)
        check(f"SSE parser, {name}: 200 random chunkings agree and equal {expect!r}", same and whole == expect)

    try:
        import numpy as np
        xs = [rng.random() for _ in range(101)]
        check("percentile matches numpy (p50,p99)", all(abs(percentile(xs, q) - float(np.percentile(xs, q * 100))) < 1e-12
                                                         for q in (.5, .99)))
    except ImportError:
        check("percentile matches numpy (skipped: numpy missing)", True)

    pts = [dict(c=rng.random(), l=rng.random()) for _ in range(200)]
    kept = pareto(pts, ["c", "l"])
    dom_ = lambda j, i: (pts[j]["c"] <= pts[i]["c"] and pts[j]["l"] <= pts[i]["l"] and
                         (pts[j]["c"] < pts[i]["c"] or pts[j]["l"] < pts[i]["l"]))
    nd = all(not any(dom_(j, i) for j in range(200)) for i in kept)
    dm = all(any(dom_(j, i) for j in range(200)) for i in range(200) if i not in kept)
    check(f"Pareto frontier correct on 200 random points ({len(kept)} kept)", nd and dm)

    # bootstrap coverage: median (n=120) and p99 at two sample sizes. The p99 numbers are INFORMATIONAL for n=160.
    def coverage(n, q, truth, trials, B, gen):
        c = 0
        for sd in range(trials):
            r = random.Random(500 + sd)
            smp = [gen(r) for _ in range(n)]
            lo, hi = bootstrap_ci(smp, lambda v: percentile(v, q), B=B, seed=sd)
            c += lo <= truth <= hi
        return c / trials
    sig = 0.6
    ln = lambda r: math.exp(r.gauss(0, sig))
    c_med = coverage(120, .5, 1.0, 150, 200, ln)
    c_160 = coverage(160, .99, math.exp(sig * 2.3263479), 150, 200, ln)
    c_1000 = coverage(1000, .99, math.exp(sig * 2.3263479), 100, 150, ln)
    print(f"      bootstrap coverage of true value (nominal 0.95): median n=120 -> {c_med:.2f}; "
          f"p99 n=160 -> {c_160:.2f}; p99 n=1000 -> {c_1000:.2f}")
    check("bootstrap median coverage within 0.88-0.995", 0.88 <= c_med <= 0.995)
    check("bootstrap p99 coverage at n=1000 is at least 0.85 and not worse than at n=160 (minus noise)",
          c_1000 >= 0.85 and c_1000 >= c_160 - 0.05)

    # summarize: failures count against SLO attainment, warm-up discarded
    recs = [Rec(intended=float(i), sent=float(i), first=i + 0.1, end=i + 0.5, ntok=11, ok=(i % 10 != 0)) for i in range(40)]
    s = summarize(recs, warmup=0)
    check("summarize counts failures (4 of 40); attainment uses ALL requests", s["failures"] == 4 and abs(s["slo_attain"] - 36 / 40) < 1e-9)
    check("warm-up discards the first 10 requests by arrival", summarize(recs, warmup=10)["n"] == 30)

    # pooled runs: warm-up must come out of EACH run, and the gap between runs must not count as wall time
    def mk(offset):
        return [Rec(intended=offset + i * 0.1, sent=offset + i * 0.1, first=offset + i * 0.1 + 0.05,
                    end=offset + i * 0.1 + 0.2, ntok=11, ok=True) for i in range(50)]
    r1, r2 = mk(0.0), mk(1000.0)                                    # run 2 starts 1000 s later
    sp = summarize_runs([r1, r2], warmup=10)
    wall_expected = 2 * ((49 * 0.1 + 0.2) - 10 * 0.1)               # per-run window after dropping 10 warm-ups
    check("pooled summary: 40 per run kept (80 total), warm-up taken from each run", sp["n"] == 80)
    check("pooled throughput uses the sum of run windows, not the 1000 s gap",
          abs(sp["tok_s"] - 80 * 11 / wall_expected) < 1e-6)
    rows = [dict(rate=r, slo_attain=a) for r, a in ((10, 1.0), (20, 0.995), (30, 0.7), (40, 0.4))]
    check("knee = highest rate with attainment >= 0.99", knee(rows) == 20)
    check("make_payload is a valid OpenAI-style streaming request",
          json.loads(make_payload(Target(model="m"), 3, 5)) == {"model": "m", "prompt": "hello hello hello", "max_tokens": 5,
                                                              "temperature": 0, "stream": True,
                                                              "stream_options": {"include_usage": True}, "ignore_eos": True})
    return all(results)


async def async_selftests():
    """Tests that need sockets or streams: chunked and close-delimited bodies, errors, truncation, merged events, timeout."""
    results = []

    def check(name, cond):
        results.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # read_body_chunks on synthetic readers (no sockets)
    r = asyncio.StreamReader()
    r.feed_data(b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"); r.feed_eof()
    got = b"".join([c async for c in read_body_chunks(r, {"transfer-encoding": "chunked"})])
    r2 = asyncio.StreamReader(); r2.feed_data(b"hello world"); r2.feed_eof()
    got2 = b"".join([c async for c in read_body_chunks(r2, {})])
    check("read_body_chunks: chunked body and close-delimited body both decode", got == got2 == b"hello world")

    async def one(prof, n_out=20, timeout=5.0):
        eng = MockEngine(prof); await eng.start()
        rec = Rec(intended=time.perf_counter())
        await one_request(Target(port=eng.port), 16, n_out, rec, timeout=timeout)
        await eng.stop()
        return rec

    base = dict(base_ms=2, per_seq_ms=0, capacity=4)
    r = await one(EngineProfile("ok", **base))
    check("normal request succeeds with 20 tokens", r.ok and r.ntok == 20 and r.nevents == 20)
    r = await one(EngineProfile("nochunk", chunked=False, **base))
    check("close-delimited (non-chunked) body succeeds", r.ok and r.ntok == 20)
    r = await one(EngineProfile("merged", merge=4, **base))
    check("merged events (4 tokens/event): success judged by server usage, not event count",
          r.ok and r.ntok == 20 and r.nevents == 5)
    r = await one(EngineProfile("err", error_rate=1.0, **base))
    check("HTTP 503 is recorded as a failure with its status", (not r.ok) and "503" in r.err)
    r = await one(EngineProfile("trunc", truncate_prob=1.0, **base))
    check("truncated stream (no [DONE]) is a failure", (not r.ok) and "incomplete" in r.err)
    r = await one(EngineProfile("slow", base_ms=50, per_seq_ms=0, capacity=4), n_out=20, timeout=0.2)
    check("client timeout is recorded as a failure", (not r.ok) and r.err == "timeout")

    # error injection flows through to the SLO-attainment denominator
    eng = MockEngine(EngineProfile("flaky", error_rate=0.3, **base), seed=3); await eng.start()
    recs = await run_open_loop(eng.port, rate=50, n_req=60, seed=2, n_out_med=10)
    await eng.stop()
    s = summarize(recs, warmup=0, slo_ttft=5.0, slo_tpot=5.0)
    check(f"with 30% injected errors: {s['failures']}/60 failures counted and attainment = {s['slo_attain']:.2f} < 0.85",
          s["failures"] > 0 and s["slo_attain"] < 0.85)
    return all(results)


# =============================================================== 7. coordinated-omission demonstrations
async def demo_coordinated_omission():
    """All numbers vary run to run (wall-clock asyncio). The assertions test qualitative relations only."""
    prof = EngineProfile("narrow", base_ms=6, per_seq_ms=0.5, capacity=4)
    # capacity ~ 4 slots / (~52 tokens * ~8 ms + prefill) ~ 9 req/s. Below it: stable. Above it: no steady state.
    out = {}
    for label, rate, n in (("below capacity", 5, 90), ("2x overload, short run", 18, 40), ("2x overload, long run", 18, 160)):
        eng = MockEngine(prof); await eng.start()
        out[label] = summarize(await run_open_loop(eng.port, rate=rate, n_req=n, seed=5), warmup=3)
        await eng.stop()
    eng = MockEngine(prof); await eng.start()
    closed = summarize(await run_closed_loop(eng.port, concurrency=4, n_req=90, seed=5), warmup=3)
    await eng.stop()
    print("  open-loop, Poisson arrivals, capacity of the mock server about 9 req/s:")
    for label, s in out.items():
        print(f"    {label:24s}: ttft_p99 = {s['ttft_p99']*1e3:8.1f} ms   e2e_p99 = {s['e2e_p99']*1e3:8.1f} ms   (n={s['n']})")
    print(f"  closed-loop, 4 workers    : ttft_p99 = {closed['ttft_p99']*1e3:8.1f} ms   e2e_p99 = {closed['e2e_p99']*1e3:8.1f} ms   (n={closed['n']})")
    # client-side stall: measuring from SEND time hides the stall; from INTENDED time exposes it
    eng2 = MockEngine(EngineProfile("wide", base_ms=4, per_seq_ms=0.1, capacity=64)); await eng2.start()
    recs = await run_open_loop(eng2.port, rate=40, n_req=100, seed=6, stall_at=40, stall_s=0.6)
    await eng2.stop()
    ok = [r for r in sorted(recs, key=lambda r: r.intended)[5:] if r.ok]
    share_send = sum(1 for r in ok if r.first - r.sent > 0.3) / len(ok)
    share_int = sum(1 for r in ok if r.first - r.intended > 0.3) / len(ok)
    m_send = sum(r.first - r.sent for r in ok) / len(ok); m_int = sum(r.first - r.intended for r in ok) / len(ok)
    print("  0.6 s client stall at request 40 (of 100, 40 req/s offered):")
    print(f"    share with TTFT > 300 ms measured from SEND time     = {share_send:5.1%}")
    print(f"    share with TTFT > 300 ms measured from INTENDED time = {share_int:5.1%}")
    print(f"    mean TTFT: from send {m_send*1e3:6.1f} ms   from intended {m_int*1e3:6.1f} ms")
    below, short, long_ = out["below capacity"], out["2x overload, short run"], out["2x overload, long run"]
    checks = {
        "below capacity, open-loop TTFT p99 is within 30x of closed-loop (a real but bounded gap)":
            below["ttft_p99"] < 30 * max(closed["ttft_p99"], 0.02),
        "under overload, open-loop TTFT p99 is at least 5x the closed-loop figure":
            long_["ttft_p99"] > 5 * max(closed["ttft_p99"], 0.02),
        "under overload, p99 keeps growing with run length (no steady state)": long_["ttft_p99"] > short["ttft_p99"],
        "stall: intended-time measurement counts more slow requests than send-time": share_int > share_send,
        "stall: intended-time mean TTFT exceeds send-time mean TTFT": m_int > m_send,
    }
    for name, c in checks.items():
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    return all(checks.values())


# =============================================================== 8. mock three-engine teardown
WARMUP = 10


async def teardown(rates=(15, 35, 70), repeats=3, n_req=150):
    profiles = [EngineProfile("engine-A", 4.0, 0.15, 16, gpus=2),
                EngineProfile("engine-B", 3.0, 0.35, 32, gpus=2),
                EngineProfile("engine-C", 5.0, 0.08, 8, gpus=1)]
    USD = 4.0
    table = {p.name: [] for p in profiles}
    for p in profiles:
        for rate in rates:
            runs = []
            for rep in range(repeats):
                eng = MockEngine(p, seed=rep); await eng.start()
                runs.append(await run_open_loop(eng.port, rate, n_req, seed=1000 * rep + rate))
                await eng.stop()
            row = summarize_runs(runs, warmup=WARMUP, usd_per_gpu_hour=USD, gpus=p.gpus)
            row.update(rate=rate, run_p99=[summarize(rs, warmup=WARMUP)["ttft_p99"] for rs in runs])
            table[p.name].append(row)
    return profiles, table


def print_teardown(profiles, table):
    print("\n  SAMPLE OUTPUT (wall-clock timing: your numbers will differ, and borderline verdicts can flip)")
    print("  engine     rate     n  ttft_p99(ms)  [95% CI]            tpot_p99(ms)  tok/s  fail  attain  $/Mtok  per-run p99 (ms)")
    for p in profiles:
        for row in table[p.name]:
            lo, hi = bootstrap_ci(row["ttft_samples"], lambda v: percentile(v, .99), B=800)
            rr = " / ".join(f"{x*1e3:.0f}" for x in row["run_p99"])
            print(f"  {p.name}  {row['rate']:4d}  {row['n']:4d}  {row['ttft_p99']*1e3:11.1f}  [{lo*1e3:6.1f},{hi*1e3:7.1f}]  {row['tpot_p99']*1e3:11.1f}  "
                  f"{row['tok_s']:5.0f}  {row['failures']:4d}  {row['slo_attain']:6.2f}  {row['usd_per_mtok']:6.2f}  {rr}")
    print("\n  knee (highest offered rate with >=99% SLO attainment, failures counted):")
    for p in profiles:
        print(f"    {p.name}: {knee(table[p.name])} req/s")
    mid = sorted({r['rate'] for r in table['engine-A']})[1]
    a = [r for r in table['engine-A'] if r['rate'] == mid][0]["ttft_samples"]
    b = [r for r in table['engine-B'] if r['rate'] == mid][0]["ttft_samples"]
    lo, hi = diff_ci(a, b, lambda v: percentile(v, .99))
    verdict = "engines differ beyond resampling noise" if lo > 0 or hi < 0 else "NOT distinguishable at this sample size"
    print(f"\n  p99 TTFT difference A-B at {mid} req/s: 95% CI [{lo*1e3:.1f}, {hi*1e3:.1f}] ms -> {verdict}")
    pts, labels = [], []
    for p in profiles:
        for row in table[p.name]:
            if row["slo_attain"] >= 0.99:
                pts.append(dict(cost=row["usd_per_mtok"], ttft=row["ttft_p99"], tput=row["tok_s"]))
                labels.append(f"{p.name}@{row['rate']}")
    kept = pareto(pts, ["cost", "ttft"], ["tput"])
    print("  SLO-feasible Pareto-optimal (cost, p99 TTFT, throughput):", ", ".join(labels[i] for i in kept))
    print("  excluded as infeasible (SLO attainment < 99%):",
          ", ".join(f"{p.name}@{r['rate']}" for p in profiles for r in table[p.name] if r["slo_attain"] < 0.99))
    cheapest = min((r["usd_per_mtok"], p.name, r["rate"]) for p in profiles for r in table[p.name])
    print(f"  note: the lowest $/Mtok anywhere is {cheapest[1]}@{cheapest[2]} ({cheapest[0]:.2f}); "
          f"it is {'feasible' if cheapest[1] + '@' + str(cheapest[2]) in labels else 'NOT SLO-feasible'}")


# =============================================================== 9. command-line client for real servers
async def run_real(args):
    target = Target.from_url(args.url, args.model, ignore_eos=not args.no_ignore_eos)
    recs = await run_open_loop(target, args.rate, args.n, args.seed, n_in_med=args.n_in, n_out_med=args.n_out)
    s = summarize(recs, warmup=args.warmup, slo_ttft=args.slo_ttft, slo_tpot=args.slo_tpot,
                  usd_per_gpu_hour=args.usd_per_gpu_hour, gpus=args.gpus)
    s.pop("ttft_samples")
    print(json.dumps(s, indent=2))
    errs = {}
    for r in recs:
        if not r.ok:
            errs[r.err.split(":")[0]] = errs.get(r.err.split(":")[0], 0) + 1
    print("failure reasons:", errs or "none")
    if args.out:
        with open(args.out, "w") as f:
            for r in recs:
                f.write(json.dumps(asdict(r)) + "\n")
        print(f"raw per-request records written to {args.out}")
    print(json.dumps(manifest(dict(url=args.url, model=args.model, rate=args.rate, n=args.n, seed=args.seed)), indent=2))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        ap = argparse.ArgumentParser(prog="ch36_benchmark_teardown.py run",
                                     description="Open-loop load against a real OpenAI-compatible /v1/completions server")
        ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
        ap.add_argument("--rate", type=float, required=True, help="offered requests per second (Poisson)")
        ap.add_argument("--n", type=int, default=600); ap.add_argument("--warmup", type=int, default=50)
        ap.add_argument("--seed", type=int, default=1)
        ap.add_argument("--n-in", type=int, default=256, help="median prompt length in words (about tokens)")
        ap.add_argument("--n-out", type=int, default=128, help="median output tokens (lognormal spread)")
        ap.add_argument("--slo-ttft", type=float, default=0.5); ap.add_argument("--slo-tpot", type=float, default=0.05)
        ap.add_argument("--usd-per-gpu-hour", type=float, default=None); ap.add_argument("--gpus", type=int, default=1)
        ap.add_argument("--no-ignore-eos", action="store_true", help="do not send ignore_eos (outputs may stop early)")
        ap.add_argument("--out", default=None)
        asyncio.run(run_real(ap.parse_args(sys.argv[2:])))
        return
    print("== self-tests ==")
    ok = selftests()
    print("== self-tests needing sockets (mock server on localhost) ==")
    ok &= asyncio.run(async_selftests())
    if "--selftest" in sys.argv:
        sys.exit(0 if ok else 1)
    print("\n== coordinated omission demonstrations (mock servers; SAMPLE OUTPUT, varies per run) ==")
    ok &= asyncio.run(demo_coordinated_omission())
    print("\n== mock three-engine teardown (engines are INVENTED; shows the method, not any real result) ==")
    profiles, table = asyncio.run(teardown())
    print_teardown(profiles, table)
    print("\n== run manifest ==")
    print(json.dumps(manifest(dict(note="mock run")), indent=2))
    print("\nALL PASS" if ok else "\nFAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

<!-- END COMPLETE INFERENCE SOURCE -->

## Chapter Summary

A defensible serving benchmark controls precision, parallelism, features, scheduling limits, output length and hardware state, and then uses one open-loop client for every engine. Open-loop arrivals with latency measured from the intended send time avoid coordinated omission: a mock server below capacity shows a bounded gap between open-loop and closed-loop clients (about tenfold in the sample run), above capacity it has no steady state at all (p99 grows with run length), and a 0.6 s client stall is visible only when measured from intended time. Success means a finished stream with the server-reported token count equal to the request, and failures stay in every denominator. Warm-up is discarded per run and throughput is computed over the sum of run windows. Percentile bootstrap intervals are reliable for medians but under-cover p99 (about 77 percent at 160 requests and 85 percent at 1,000 against a nominal 95), so p99 claims need at least 1,000 requests, repeats and agreement between runs; in the mock study the comparison of two engines sat near the noise floor and flipped between executions at small sample sizes. Engines are ranked by an SLO-feasible Pareto frontier and by the knee where attainment drops below 99 percent, never by cost per token or peak throughput alone: the cheapest point in the mock table failed its SLO. The harness is verified on CPU against mock servers; the three real engines still need to be measured on H100s following the runbook.

## Self-Check Questions

1. Why does a closed-loop load generator understate latency on an overloaded server?
2. Why is latency measured from the intended send time and not the actual one?
3. Why must failed requests remain in the denominator of SLO attainment?
4. In Worked Example 36.8, engine C at 35 req/s has the lowest cost per million tokens. Why is it not a candidate, and what else in the table is cheaper than every feasible point?
5. An open-loop run at 18 req/s on a server with 9 req/s capacity reports p99 TTFT of 10 s after 157 requests and 2.3 s after 37. What should the report say?
6. Why must each engine be run alone on the machine?
7. Why does `--request-rate inf` not measure steady-state serving?
8. List three configuration differences that would make an engine comparison invalid even with a perfect load generator.
9. Why does the client use the server's `usage.completion_tokens` and not the number of streamed events?

## Worked Solutions

1. The client sends a new request only after a reply returns, so when the server slows down the client sends less, the queue never builds as it would for independent users, and measured latency stays low.
2. If the client stalls, requests that should have been sent are sent late. Measuring from the actual send time makes them look fast and hides the stall; measuring from the intended time charges the delay, as a real user would experience it.
3. Dropping failures lets an overloaded server look better by removing its worst outcomes. Counting them as misses makes attainment reflect what users received.
4. Only 21 percent of its requests meet the SLO, so it is far past its knee. Engine A at 70 req/s (cost 1.12) is also cheaper than every feasible point (the cheapest feasible is 1.41 at engine A, 35 req/s) and also infeasible (45 percent attainment). Cheap points above the knee are cheap because the GPUs are saturated, which is the cause of the latency failure.
5. It should say the system is unstable at that load: latency grows with the length of the run, so no p99 is meaningful, and capacity is below 18 req/s. Report the knee from rates below capacity instead.
6. Engines compete for GPU memory, clocks, PCIe and CPU, so a second engine running changes the first's results and the comparison measures interference.
7. Every request is issued at time zero, so the server faces a one-time burst and then drains it; there is no steady arrival process and the latency reflects queue position, not service under load.
8. Examples: different precision or KV-cache dtype, different tensor-parallel degree, prefix caching on for one engine and off for another, speculative decoding on for one, different maximum batch or token limits, different output lengths because one engine stops early.
9. Real engines can send several tokens in one event or send empty deltas, so event counts under- or over-count tokens and would mark successful requests as failures. The server's own count is the work it reports having done; content-event count remains a diagnostic only. Missing server usage causes a benchmark failure rather than an unverifiable token-count claim.

## Coding Exercises

1. **Run-level bootstrap.** Write `run_bootstrap_ci(runs, stat, B)` that resamples whole runs (with replacement) and then requests within each chosen run, and compare its coverage of the true p99 with the plain bootstrap in the self-test. Expect it to be wider; with only three runs it is a poor estimate itself, which is why more repeats are the real remedy.
2. **Prefix-cache probe.** Extend `make_payload` to send either unique random prompts or a fixed prompt, and write a test with a mock server that responds faster for repeated prompts, showing that the fixed-prompt run reports lower TTFT than the unique-prompt run on an identical engine.
3. **Closed-loop with a think-time.** Add a `think_s` parameter to `run_closed_loop` and show that with a long think time the closed-loop client approaches open-loop behaviour at low load, and explain why it still cannot reveal overload.

### Solution sketches

1. Build `runs` as lists of per-request values; for each of B iterations draw `len(runs)` run indices with replacement, then for each chosen run draw its requests with replacement, concatenate, and apply the statistic. Run the coverage experiment with 5 simulated runs of 200 requests each.
2. The mock handler uses `hash(prompt)` in a set to decide whether to apply a cache discount to the prefill time; the test sends 40 identical prompts and 40 unique ones and asserts the p50 TTFT difference exceeds the discount minus noise.
3. After each reply the worker does `await asyncio.sleep(think_s)` before its next request. As `think_s` grows, offered load per worker falls and queues vanish, so latency looks like the open-loop result at low load; at overload the workers still wait for replies, so the offered load stays capped at the number of workers divided by the (growing) response time.

## Sources

- vLLM `bench serve` command reference: <https://docs.vllm.ai/cli/bench/serve.html>
- SGLang bench serving guide: <https://docs.sglang.io/docs/developer_guide/bench_serving.md>
- SGLang benchmark and profiling guide: <https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling>
- TensorRT-LLM, running benchmarks with `trtllm-serve`: <https://nvidia.github.io/TensorRT-LLM/1.3.0rc29/commands/trtllm-serve/run-benchmark-with-trtllm-serve.html>
- NVIDIA developer blog, LLM inference benchmarking with TensorRT-LLM (`trtllm-bench`): <https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm>
- DistServe (goodput definition): <https://arxiv.org/pdf/2401.09670>
