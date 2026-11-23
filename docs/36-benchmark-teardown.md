# Chapter 36: Public Benchmark Teardown: vLLM, SGLang and TensorRT-LLM

!!! abstract "Goal and prerequisites"

    Build a benchmark you could defend in public. You will write a load generator and statistics library that avoid the classic measurement errors, test it against mock servers, and receive a runbook for pointing it at vLLM, SGLang and TensorRT-LLM on real H100s. Read [Chapter 32](32-inference-engine-and-harness.md) (metrics and the simulated engine) first; this chapter replaces its simulator with a real network client.

!!! warning "Evidence level: read this first"

    - **Verified here (CPU):** the client, the streaming parsers, the percentile and bootstrap code, the Pareto and knee logic, the run manifest, and a mock three-engine study. Seven self-tests pass.
    - **Invented:** the three "engines" in the mock study are made-up servers with made-up parameters. **No number in this chapter is a vLLM, SGLang or TensorRT-LLM result**, and the mock study must not be quoted as one.
    - **Untested on H100:** every launch command in Section 36.9 and the real teardown itself. Flags change between versions; confirm each against `--help` for the exact version you run.

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

The rest of the chapter is a defence against this list.

## 36.2 What each engine ships

All three engines expose an OpenAI-compatible HTTP server and ship their own benchmark client. That is both a convenience and a trap, because each client has its own defaults.

| Engine | Server | Vendor benchmark client | Notes from the documentation |
|---|---|---|---|
| vLLM | `vllm serve` | `vllm bench serve` | `--request-rate` with Poisson arrivals by default; `inf` sends everything at time zero; `--burstiness` switches to a gamma process; `--max-concurrency` caps in-flight requests; `--metadata` and `--save-detailed` record context and per-request data; goodput is defined in the DistServe paper's terms (ttft, tpot, e2el) |
| SGLang | `python -m sglang.launch_server` | `python3 -m sglang.bench_serving` | Reports TTFT, TPOT and inter-token latency; its documentation says to start each server on the same hardware and precision and **run one at a time**; `bench_one_batch_server` sends a single batch and is explicitly not steady-state |
| TensorRT-LLM | `trtllm-serve` | `tensorrt_llm.serve.scripts.benchmark_serving` and `trtllm-bench` | Serving is configured with flags and a YAML `--config`; the project's own methodology is launch with `trtllm-serve`, then run the benchmark script |

**Design decision:** use **one** client for all three engines, written once, against the OpenAI-compatible endpoint, with the same tokenizer counting tokens. Use each vendor's client only as a cross-check: on the same server the numbers should agree within noise. If they do not, find out why before publishing anything.

## 36.3 Experimental design

Decide these before running anything, and write them in the report.

| Variable | Rule | Why |
|---|---|---|
| Model and revision | Identical weights, identical checkpoint revision | |
| Precision | Same weight and KV-cache dtype, or report each engine's best supported option as a separate, labelled row | An FP8 engine against a BF16 engine is two experiments |
| Parallelism | Same GPU count and tensor-parallel degree | |
| Batch and scheduling limits | Equalize the maximum sequences and batched tokens where the engines expose them; otherwise report the values | Dominant drivers of the throughput and latency trade-off |
| Features | Prefix caching, speculative decoding, chunked prefill: all explicit, all reported | |
| Output length | Force a fixed output length per request (an `ignore_eos`-style option), drawn from a stated distribution | Equal work per request |
| Prompts | Fixed seed, same prompt set, length counted with one tokenizer | |
| Sampling | Same temperature, top-p and seed settings | |
| Load | Open-loop Poisson arrivals at several rates, spanning idle to overloaded | Section 36.5 |
| Duration | Long enough for hundreds of requests per point after warm-up | Percentiles need samples |
| Repeats | At least two, ideally five, at each point, restarting the server | Run-to-run variation |
| Hardware state | Record GPU model, driver, CUDA version, clocks, power cap, container tag | Reproducibility |
| Isolation | Server and client on separate processes (ideally separate hosts or cores); one engine on the machine at a time | |

## 36.4 The client, piece by piece

The full file is `ch36_benchmark_teardown.py`. [Download it](examples/inference-systems/ch36_benchmark_teardown.py). It uses only the Python standard library for the mock study; for real servers, keep the same logic but you may prefer an HTTP library with connection pooling.

### Streaming parsers

Servers stream tokens as Server-Sent Events, usually inside HTTP chunked encoding. The two layers can split an event anywhere, so the parser must not assume event boundaries line up with network reads. The parser accumulates bytes and emits only complete events:

```python
class SSEParser:
    def feed(self, chunk: bytes):
        self.buf += chunk
        out = []
        while True:
            i = self.buf.find(b"\n\n")
            if i < 0:
                break
            event, self.buf = self.buf[:i], self.buf[i + 2:]
            for line in event.split(b"\n"):
                line = line.rstrip(b"\r")
                if line.startswith(b"data:"):
                    out.append(line[5:].strip().decode())
        return out
```

Test: the same stream cut at 200 random places yields identical events. A parser that fails this test will mis-time tokens whenever a packet boundary falls inside an event.

### Open-loop arrivals and intended time

Arrival times are drawn in advance from a Poisson process (exponential gaps). Each request records when the schedule said to send it (`intended`) and when it was actually sent (`sent`). Latency is measured from `intended`:

```python
rec = Rec(intended=t0 + at)
...
await asyncio.sleep(delay)                      # wait until the intended time
tasks.append(asyncio.create_task(one_request(port, ni, no, rec)))
```

If the client itself stalls (garbage collection, a busy event loop, a slow DNS lookup), the delay is charged to the request rather than hidden.

### What counts as success

A request is successful only if the stream ended with the terminal `[DONE]` event **and** delivered exactly the requested number of tokens. A truncated stream, an HTTP error, a timeout or an exception is a failure, recorded with its reason. Failures stay in the denominator of every rate and attainment figure.

```python
rec.ok = done and rec.ntok == n_out
```

### Timestamps

`first` is the arrival time of the first streamed token, `end` the arrival time of the last event, and `itl` the gaps between consecutive tokens. TTFT is `first - intended`, TPOT is `(end - first) / (ntok - 1)`, end-to-end latency is `end - intended`.

Note that a "token" here is one streamed content event. Some servers batch several tokens into one event, which makes inter-token latency look larger and TPOT unchanged. Check how each engine streams, and count output tokens with the same tokenizer on the received text when exact counts matter.

## 36.5 Coordinated omission, demonstrated

Suppose a server is overloaded. A closed-loop client with N workers sends a new request only when a reply comes back, so it automatically sends **less** when the server is slow. The measured latency stays modest because the client never builds the queue a real population of users would. An open-loop client keeps the arrival schedule, and the queue grows as it would in production.

### Worked Example 36.5.1

A mock server with capacity for 4 concurrent sequences, offered 20 requests per second open-loop, against 4 closed-loop workers sending the same prompts:

```
  open-loop  @20 rps offered: ttft_p99=  4993.7 ms   e2e_p99=  5653.7 ms
  closed-loop, 4 workers    : ttft_p99=    45.0 ms   e2e_p99=  1059.0 ms
```

The same server looks like it has a 45 ms p99 time to first token or nearly 5 seconds, depending on the load generator. The open-loop number is the honest one for an offered load of 20 requests per second; the closed-loop number answers a different question (how fast is it with 4 users who always wait?).

### Worked Example 36.5.2: a stalled client

The client blocks its own event loop for 0.6 s while 100 requests are being issued at 40 per second. About 24 requests should have been sent during the stall; they are sent late, immediately afterwards.

```
  0.6 s client stall at request 40 (of 100, 40 req/s offered):
    share of requests with TTFT > 300 ms, measured from SEND time     =  3.2%  (only requests already in flight)
    share of requests with TTFT > 300 ms, measured from INTENDED time = 12.6%  (also those that should have been sent during the stall)
    mean TTFT: from send   36.7 ms   from intended   92.6 ms
```

Measured from the send time, the late requests look perfectly healthy and the benchmark's own stall disappears from the data. Measured from the intended time, the stall is counted, which is what a user who clicked at the scheduled moment would have experienced. (This also shows why a benchmark client must be lightweight and isolated: a stalled client is charged to the server under test.)

## 36.6 Statistics

### Warm-up

The first requests hit cold caches, compile kernels or capture CUDA graphs. Discard them **by arrival order**, a fixed count chosen in advance, and report the number discarded. The self-test checks that `warmup=10` removes exactly the first ten.

### Percentiles

Use linear interpolation between order statistics (the default of common numerical libraries). The self-test compares the implementation with NumPy for p50 and p99. Report p50, p99 and, where the sample allows, p99.9; report the sample size next to every percentile, because the p99 of 50 requests is a single observation.

### Bootstrap confidence intervals

Resample the per-request values with replacement many times, compute the statistic each time, and take the 2.5th and 97.5th percentiles. The self-test verifies the method: over 150 synthetic datasets with a known median, the 95 percent interval contained the truth in 96 percent of them.

```python
def bootstrap_ci(xs, stat, B=800, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    vals = sorted(stat([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return vals[int(B * alpha / 2)], vals[int(B * (1 - alpha / 2)) - 1]
```

!!! warning "What this interval does and does not cover"

    Resampling requests captures sampling noise within the runs you did. It does **not** capture run-level effects (a noisy neighbour, a thermal throttle, a different server start-up). That is why every point is repeated with a fresh server and the run-to-run spread is printed beside the interval.

### Comparing two engines

To ask "does A differ from B?", bootstrap the **difference** of the statistic. If the interval excludes zero the difference exceeds resampling noise **for this workload**; if it includes zero the data cannot separate them.

## 36.7 Pareto frontiers, knees and cost

One number cannot rank engines: they trade latency, throughput and cost, and the trade changes with load.

- **Pareto frontier:** a point is dominated if another is no worse on every axis and better on at least one. The frontier is what remains. The implementation is tested on 200 random points: every kept point is non-dominated, and every dropped point is dominated.
- **Feasibility first:** only points that meet the SLO belong on the frontier. A cheap point at which 40 percent of requests miss their latency target is not a competitor, it is a failure.
- **Knee:** the highest offered rate at which SLO attainment stays at or above a threshold (here 99 percent, with failures counted as misses). Capacity planning uses the knee, not peak throughput.
- **Cost:** `usd_per_gpu_hour x GPUs x wall_hours / million_output_tokens`. State the price and where it comes from; an assumed price makes the cost column an assumption.

## 36.8 A mock teardown, end to end

Three **invented** engines (different per-token latency, concurrency capacity and GPU count) are served by mock servers and driven at 15, 35 and 70 requests per second, two repeats each. Everything below demonstrates the method.

```
  engine     rate  ttft_p99(ms)  [95% CI]            tpot_p99(ms)  tok/s  fail  attain  $/Mtok  run-to-run p99 (ms)
  engine-A    15         37.7  [  32.8,   39.2]          6.1    689     0    1.00    3.23  37 / 39
  engine-A    35         59.7  [  47.4,   68.8]          7.1   1474     0    1.00    1.51  44 / 68
  engine-A    70        647.1  [ 625.5,  657.9]          7.2   1848     0    0.81    1.20  654 / 616
  engine-B    15         37.2  [  32.3,   39.8]          7.3    694     0    1.00    3.20  37 / 39
  engine-B    35         44.7  [  33.2,   51.8]         12.3   1445     0    1.00    1.54  36 / 46
  engine-B    70        619.8  [ 524.6,  626.0]         14.9   1755     0    0.94    1.27  623 / 400
  engine-C    15        155.6  [  87.3,  200.1]          6.5    681     0    1.00    1.63  70 / 165
  engine-C    35       1155.0  [1068.5, 1229.9]          7.5   1070     0    0.42    1.04  1097 / 1187
  engine-C    70       2421.7  [2236.8, 2429.3]          6.6   1054     0    0.19    1.05  2425 / 2200

  knee (highest offered rate with >=99% SLO attainment, failures counted):
    engine-A: 35 req/s
    engine-B: 35 req/s
    engine-C: 15 req/s

  p99 TTFT difference A-B at 35 req/s: 95% CI [0.4, 31.2] ms -> engines differ beyond resampling noise
  SLO-feasible Pareto-optimal (cost, p99 TTFT, throughput): engine-A@35, engine-B@15, engine-B@35
  excluded as infeasible (SLO attainment < 99%): engine-A@70, engine-B@70, engine-C@35, engine-C@70
```

How to read it, and what each line teaches:

1. **Intervals matter.** Engine C at 15 req/s has a p99 TTFT of 156 ms with an interval of 87 to 200 ms, and its two runs gave 70 and 165 ms. A single run could have reported either, and the engine would have looked twice as good or bad.
2. **The comparison is borderline, and unstable.** The A-minus-B interval at 35 req/s is `[0.4, 31.2]`, just excluding zero. An earlier execution of the same program gave `[-0.6, 30.7]`, which includes zero and flipped the verdict to "not distinguishable". Same code, same mock servers, different result: the data sit at the edge of what 180 requests can resolve. The right response is more requests and more repeats, not choosing the favourable run. This is the single most important habit in the chapter.
3. **Throughput and latency disagree.** At 70 req/s engine A has the highest throughput (1,848 tokens per second) and the lowest cost per token in the table, yet only 81 percent of its requests meet the SLO. Cost per token alone would crown the point that is failing users.
4. **Different knees.** A and B sustain 35 req/s within SLO; C only 15. An engine's capacity is where it stops meeting the SLO, which is not where its throughput peaks.
5. **Failures are zero here** because the mock servers have no injected errors. Set `error_rate` on a profile to see them counted against attainment.

The mock engines were built to produce these shapes. Real engines will produce different ones, and some of the differences will be configuration, not engine quality.

## 36.9 Runbook for real H100s (untested)

!!! warning "UNTESTED ON H100"

    None of this was executed. Flags below follow each project's documentation at the time of writing but change between releases; run `--help` for your version and record the version in the report.

### 1. Capture the environment first

```bash
nvidia-smi -q > env/nvidia-smi-full.txt
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit --format=csv > env/gpu.csv
python3 -c "import platform,sys;print(platform.platform(),sys.version)" > env/host.txt
pip freeze > env/pip-freeze.txt            # inside each engine's container or venv
```

Pin each engine by container image tag or exact package version. Lock the GPU clocks or at least record them, and record whether other processes share the GPU.

### 2. Launch one engine at a time, same model, same GPUs

```bash
# vLLM  (check: vllm serve --help)
vllm serve <model> --tensor-parallel-size 2 --port 8000

# SGLang  (check: python -m sglang.launch_server --help)
python -m sglang.launch_server --model-path <model> --tp-size 2 --port 30000

# TensorRT-LLM  (check: trtllm-serve --help). Settings go in a YAML file passed with --config.
trtllm-serve <model> --tp_size 2 --max_batch_size <N> --max_num_tokens <N> --max_seq_len <N> --config engine.yml
```

Equalize the scheduling limits (maximum sequences, maximum batched tokens, maximum sequence length) where each engine exposes them, and write the values in the report. Record any KV-cache dtype and quantization option explicitly.

### 3. Point the one client at each server

Run the Section 36.4 client against `http://127.0.0.1:<port>/v1/completions` with the same seed, prompt set, output-length distribution and rates. Restart the server between repeats.

### 4. Cross-check with the vendor clients

```bash
vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --dataset-name random \
    --num-prompts 1000 --request-rate 20 --metadata engine=vllm
python3 -m sglang.bench_serving --backend sglang --port 30000 --dataset-name random \
    --num-prompts 1000 --request-rate 20
python -m tensorrt_llm.serve.scripts.benchmark_serving --model <model> --backend openai ...
```

Do not use `--request-rate inf` for a steady-state comparison: in vLLM's client that sends every request at time zero, which measures draining a burst. Use `--max-concurrency` only if you intend a capped, closed-loop-style test, and label it.

### 5. Sweep and decide

Run at least five offered rates from well below to well above the expected knee, two or more repeats each, then produce the table of Section 36.8 and the report below.

## 36.10 Report template

| Section | Contents |
|---|---|
| Question | One sentence: "Which engine gives the lowest cost per million tokens at p99 TTFT under 500 ms and p99 TPOT under 50 ms for chat-length requests on 2 x H100?" |
| Environment | GPU model and count, driver, CUDA, host CPU, container tags, clocks, power cap |
| Engines and configs | Exact launch commands, scheduling limits, precision, features on and off |
| Workload | Arrival process and rates, prompt and output length distributions with seed, dataset origin, number of requests per point and warm-up discarded |
| Results | Per engine and rate: TTFT and TPOT p50 and p99 with intervals, throughput, failures, SLO attainment, cost per million tokens, run-to-run spread |
| Frontier | SLO-feasible Pareto set and the knee per engine |
| Where each engine loses | At least one workload or rate where each engine is worst, and why |
| Threats to validity | Items from Section 36.1 you could not control |
| Reproduction | Repository with scripts, raw per-request data, manifest and a single command |

## 36.11 Failure-mode catalogue

| Symptom | Likely cause | Check |
|---|---|---|
| Throughput far below the vendor's published figure | Different workload (shorter outputs, higher concurrency), batch limits, or a closed-loop client | Run the vendor's client on the same server |
| p99 TTFT swings between runs | Too few requests, noisy host, thermal or power throttling | Increase requests, record clocks, repeat |
| Zero failures at every load | Failures are being swallowed | Inject errors and confirm they are counted |
| Latency improves at high load | Closed-loop client | Switch to open-loop |
| Engines differ only in the tail | Scheduling policy differences | Compare TPOT distributions and queueing |
| One engine uses much less memory and "wins" | Smaller KV cache, so a lower concurrency ceiling | Compare maximum concurrent sequences |
| Token counts differ across engines | Different tokenizers or batched streaming events | Count tokens from the text with one tokenizer |
| Results change after a driver update | Environment drift | Re-run, and keep the manifest with the data |

## Chapter Summary

A defensible serving benchmark controls precision, parallelism, features, scheduling limits, output length and hardware state, and then uses one open-loop client for every engine. Open-loop arrivals with latency measured from the intended send time avoid coordinated omission: in a mock test the same server showed a 45 ms p99 TTFT closed-loop and almost 5 seconds open-loop, and a 0.6 s client stall raised the share of slow requests from 3.2 to 12.6 percent once measured from intended time. Failures stay in every denominator. Percentiles come with sample sizes and bootstrap intervals, repeats expose run-level noise, and a comparison is claimed only when the interval for the difference excludes zero; in the mock study the A-versus-B verdict sat on that boundary and flipped between executions, which is a reason to collect more data. Engines are ranked by an SLO-feasible Pareto frontier and by the knee where attainment drops below 99 percent, not by peak throughput or cost per token alone. The harness is verified on CPU against mock servers; the three real engines still need to be measured on H100s following the runbook.

## Self-Check Questions

1. Why does a closed-loop load generator understate latency on an overloaded server?
2. Why is latency measured from the intended send time and not the actual one?
3. Why must failed requests remain in the denominator of SLO attainment?
4. In Worked Example 36.8, engine A has the best throughput and cost per token at 70 req/s. Why is it not the recommended operating point?
5. Two engines have p99 TTFT of 59.7 and 44.7 ms and the difference interval is `[0.4, 31.2]`. What may you conclude, and what would you do next?
6. Why must each engine be run alone on the machine?
7. Why does `--request-rate inf` not measure steady-state serving?
8. List three configuration differences that would make an engine comparison invalid even with a perfect load generator.

## Worked Solutions

1. The client sends a new request only after a reply returns, so when the server slows down the client sends less, the queue never builds as it would for independent users, and measured latency stays low.
2. If the client stalls, requests that should have been sent are sent late. Measuring from the actual send time makes them look fast and hides the stall; measuring from the intended time charges the delay, as a real user would experience it.
3. Dropping failures lets an overloaded server look better by removing its worst outcomes. Counting them as misses makes attainment reflect what users received.
4. Only 81 percent of its requests meet the latency SLO, so it is past its knee. The cheapest point that fails the SLO is not a valid option; the operating point is the highest rate meeting the SLO, 35 req/s.
5. At this sample size the difference is just distinguishable from resampling noise, and its size is between roughly 0.4 and 31 ms, so the claim should be modest. Collect more requests and repeats and check whether the interval stays away from zero; an interval that flips with a rerun is not evidence.
6. Engines compete for GPU memory, clocks, PCIe and CPU, so a second engine running changes the first's results and the comparison measures interference.
7. Every request is issued at time zero, so the server faces a one-time burst and then drains it; there is no steady arrival process and the latency reflects queue position, not service under load.
8. Examples: different precision or KV-cache dtype, different tensor-parallel degree, one engine with prefix caching or speculative decoding on and the other off, different maximum batch or token limits, different output lengths because one engine stops early.

## Sources

- vLLM `bench serve` command reference: <https://docs.vllm.ai/cli/bench/serve.html>
- SGLang bench serving guide: <https://docs.sglang.io/docs/developer_guide/bench_serving.md>
- SGLang benchmark and profiling guide: <https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling>
- TensorRT-LLM, running benchmarks with `trtllm-serve`: <https://nvidia.github.io/TensorRT-LLM/1.3.0rc29/commands/trtllm-serve/run-benchmark-with-trtllm-serve.html>
- NVIDIA developer blog, LLM inference benchmarking with TensorRT-LLM (`trtllm-bench`): <https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm>
- DistServe (goodput definition): <https://arxiv.org/pdf/2401.09670>
