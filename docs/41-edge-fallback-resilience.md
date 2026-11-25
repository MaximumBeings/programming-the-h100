# Chapter 41 — Resilience and Edge Fallback (WebGPU and MLX)

!!! warning "Evidence status"
    **Run on CPU, in a simulator:** the circuit breaker, retry budget and jittered backoff; a per-client simulation of five outage shapes under six policies; a fleet simulation of a capacity dip with retries; the edge-feasibility arithmetic; 17 self-tests and three experiments (`ch41_edge_fallback.py`, standard library only). Experiment cells show the mean and, in brackets, the minimum–maximum over three seeds.
    **Not run on any device.** No WebGPU page, WebLLM engine or MLX model was executed for this chapter. The remote and edge speeds (remote 0.5 s to first token and 60 tokens/s; edge 300 tokens/s prefill, 25 tokens/s decode, 2048-token context, one request at a time with two waiting) and the fleet numbers (70 req/s offered, 100 req/s capacity) are **assumed**. The browser and Apple-silicon runbooks in 41.10 are **UNTESTED**.
    **What the simulator leaves out:**

    - **Answer quality.** The edge model is smaller, so its answers differ from the remote model's. The simulator counts an edge answer as a success; you must decide whether that is acceptable for your product.
    - **Device variance.** Thermal throttling, battery saver, background tabs, memory pressure and the device's own GPU driver are not modelled.
    - **Model download and load time.** The first edge answer on a cold device needs the weights downloaded and compiled; the simulator assumes they are already resident.
    - **Network detail.** Failures are drawn from five idealised shapes, with no TLS, DNS, proxy or mobile-network behaviour.
    - **Correlated clients.** The per-client simulation has one client. The fleet simulation has a single global queue and identical clients.

A remote GPU service fails in ways that a health check does not see: connections refused, requests that hang, responses that are ten times slower, streams that stop halfway. This chapter builds the client-side machinery that keeps the user's request alive through those failures (timeouts, bounded retries, a circuit breaker, a local fallback) and the server-side rule that keeps a recovering service from being buried by its own clients. The central results are about interactions: each mechanism helps in some failures and hurts in others, and retries can turn a 30-second dip into a permanent outage.

## 41.1 Five ways a remote endpoint fails

| Shape | What the client sees | Fast or slow failure |
|---|---|---|
| **refused** | the connection is rejected at once | fast: tens of milliseconds |
| **hang** | the request is accepted and nothing comes back | slow: the client's timeout decides |
| **brownout** | some requests exceed the timeout, others succeed | mixed |
| **flapping** | the endpoint alternates between down and up every 10 s | mixed |
| **mid-stream drop** | the first tokens arrive, then the stream ends early | slow, and the work already received is lost unless you keep it |

Each shape punishes a different policy. A fast failure makes immediate retries useless (they all fail within a second). A slow failure makes every retry expensive (3 s each). A mixed failure defeats a breaker that counts consecutive failures, because successes keep resetting the count. A mid-stream drop makes "retry from scratch" redo work that was already paid for.

The simulated timeline is 300 s with an outage from 60 s to 180 s (40% of the time). Requests arrive about every 12 s. Ten per cent have a 3000-token prompt, which is longer than the edge model's 2048-token context.

## 41.2 The building blocks

### Circuit breaker

```python
class CircuitBreaker:
    def allow(self, now):            # -> ("closed" | "probe" | None, gen)
        ...
    def record(self, gen, ok, now):  # ignores results from an older gen
        ...
```

States: **closed** (calls go through; `threshold` consecutive failures open it), **open** (calls are refused locally for `open_for` seconds), **half-open** (exactly one probe goes through; success closes, failure reopens). Two details carry the weight:

- **Generation counter.** Each state change increments `gen`. A call remembers the `gen` it started under; if the breaker has changed state since, its result is ignored. Without this, a slow success that was issued before the breaker opened would close it again.
- **Probe timeout.** If the half-open probe never reports back (the caller crashed, the request was cancelled), the breaker would wait forever. After `open_for` more seconds the probe is declared lost and a new one is allowed.

### Retry budget

```python
class RetryBudget:
    def on_request(self): tokens = min(cap, tokens + ratio)
    def try_retry(self):  # spend one token or refuse
```

Each original request earns `ratio` tokens (here 0.1); each retry spends one. Over any period the number of retries cannot exceed the initial tokens plus `ratio` times the number of requests. A retry budget turns "each client retries up to 3 times" (up to 4× the load) into "retries add at most 10% to the load".

### Backoff with full jitter

`delay = uniform(0, min(cap, base · 2^attempt))`. Exponential growth spaces retries out; the uniform draw from zero spreads clients that failed together so they do not return together. Jitter does not reduce the *amount* of retried work; it only spreads it in time (41.7 shows the limit of that).

## 41.3 Self-tests (17, all pass)

```text
python3 ch41_edge_fallback.py tests
```

| # | Property |
|---|---|
| 1 | The breaker opens after N consecutive failures; a success resets the count |
| 2 | Open refuses until `open_for`, then allows exactly one probe; success closes |
| 3 | A failed probe reopens; a lost probe is replaced after `open_for` |
| 4 | A stale result (older generation) is ignored |
| 5 | Retry budget: retries granted ≤ initial + ratio × requests, for ratios 0, 0.1, 0.5 |
| 6 | Backoff stays in [0, min(cap, base·2^k)] with mean about half the upper bound |
| 7 | No outage: every policy succeeds on every request with one remote try each |
| 8 | A fixed seed gives identical results |
| 9 | Refused: edge fallback beats a single try; the breaker cuts remote load against plain retries |
| 10 | The edge never serves beyond its context window and never exceeds its queue limit |
| 11 | The edge cannot rescue prompts longer than its context |
| 12 | After a mid-stream drop, resume gives lower mean edge latency than restarting (5 of 5 seeds) |
| 13 | Fleet without a dip: all policies complete at least 99% of requests |
| 14 | Fleet conservation: requests served ≤ capacity; completed ≤ arrivals |
| 15 | Fleet: expired-request shedding recovers sooner and wastes less than naive retries |
| 16 | Fleet retry budget: retries ≤ initial + 10% of arrivals |
| 17 | Edge arithmetic hand checks (41.8) |

## 41.4 The six client policies

| Policy | Attempts | Backoff | Breaker | Edge | Resume |
|---|---|---|---|---|---|
| P0 single try | 1 | | | | |
| P1 retry ×3, no backoff | 4 | | | | |
| P2 + jittered backoff | 4 | yes | | | |
| P3 + circuit breaker | 4 | yes | yes | | |
| P4 + edge fallback | 4 | yes | yes | yes | |
| P5 + mid-stream resume | 4 | yes | yes | yes | yes |

Every policy has the same 30 s deadline per request and a 3 s wait for the first token. **Resume** keeps the tokens already received; the next attempt (remote or edge) replays `prompt + received tokens` and generates only the remainder, so an edge fallback after a drop at token 100 of 200 pays prefill for the extra 100 tokens but decodes only 100.

### Hand checks

- **No protection.** An outage covering 40% of the timeline should leave about 60% success. P0 measured 57.3% (54.2–60.7% over seeds); with about 25 requests per client, a few requests either way moves it by 4 points.
- **Edge fallback ceiling.** The edge fails only on requests it cannot hold: 10% long prompts × 40% outage = 4%, so about 96% success. P4 measured 95.8%.
- **Remote load.** With four attempts on every request in the outage, `0.4 × 4 + 0.6 × 1 = 2.2` remote calls per request. P1 measured 2.25–2.28.

## 41.5 Experiment 1 — five outage shapes, six policies

Success rate (%), 95th-percentile latency of successful requests (s), share served by the edge (%), remote calls per request.

**Refused (fast failure)**

| Policy | Success % | p95 s | Edge % | Remote/req |
|---|---|---|---|---|
| P0 | 57.3 [54.2–60.7] | 5.3 | 0 | 1.00 |
| P1 | 57.3 [54.2–60.7] | 5.3 | 0 | 2.28 |
| P2 | 57.3 [54.2–60.7] | 5.3 | 0 | 2.28 |
| P3 | 55.8 [52.4–60.7] | 5.3 | 0 | 0.89 |
| P4 | 95.8 [95.2–96.4] | 20.5 [15.9–26.8] | 40.1 | 0.89 |
| P5 | 95.8 [95.2–96.4] | 20.5 [15.9–26.8] | 40.1 | 0.89 |

**Hang (slow failure)**

| Policy | Success % | p95 s | Edge % | Remote/req |
|---|---|---|---|---|
| P0 | 57.3 [54.2–60.7] | 5.3 | 0 | 1.00 |
| P1 | 60.3 [58.3–61.9] | 8.1 [5.1–13.6] | 0 | 2.25 |
| P3 | 55.8 [52.4–60.7] | 5.3 | 0 | 0.85 |
| P4 | 95.8 [95.2–96.4] | 23.0 [21.4–23.8] | 40.1 | 0.85 |

**Brownout (half the requests time out)**

| Policy | Success % | p95 s | Edge % | Remote/req |
|---|---|---|---|---|
| P0 | 90.9 [85.7–95.2] | 5.3 | 0 | 1.00 |
| P1–P5 | 100.0 | 7.1–7.2 | 0 | 1.14 |

**Flapping (10 s down, 10 s up)**

| Policy | Success % | p95 s | Edge % | Remote/req |
|---|---|---|---|---|
| P0 | 81.5 [76.2–89.3] | 5.2 | 0 | 1.00 |
| P1 | 98.4 [95.2–100.0] | 11.2 | 0 | 1.40 |
| P2 | 98.4 [95.2–100.0] | 11.6 | 0 | 1.40 |
| P3 | 80.8 [70.8–100.0] | 6.4 | 0 | 1.03 |
| P4 | 96.8 [90.5–100.0] | 19.1 [8.6–25.4] | 16.1 [0–29.2] | 1.03 |

**Mid-stream drop**

| Policy | Success % | p95 s | Edge % | Remote/req |
|---|---|---|---|---|
| P0 | 57.3 [54.2–60.7] | 5.3 | 0 | 1.00 |
| P1 | 60.3 [58.3–61.9] | 7.7 | 0 | 2.27 |
| P3 | 55.8 [52.4–60.7] | 5.3 | 0 | 0.88 |
| P4 | 95.8 [95.2–96.4] | 23.7 [20.4–28.7] | 40.1 | 0.88 |
| P5 | 97.4 [95.8–100.0] | 16.0 [11.7–23.4] | 41.7 | 0.88 |

(P2 equals P1 within rounding in every row except flapping and mid-stream drop, where it is slightly slower; P5 equals P4 except on mid-stream drop, where only resume has anything to save.)

### What the tables say

1. **Retries do nothing for a fast failure.** Under *refused*, P1 and P2 match P0 exactly: all four attempts fail inside one second and the outage is two minutes long. The only effect is 2.3× the remote calls.
2. **The edge is what turns a 40% outage into a 4% one.** P4 reaches 95.8% in three of the five shapes. The residual 4% is the long prompts the edge cannot hold. The price is latency: the edge's p95 is 20–24 s because each edge answer takes several seconds after the client has already spent up to 12 s on remote attempts before the breaker opens.
3. **A breaker without a fallback loses availability.** P3 is *below* P0 (55.8% against 57.3%) under refused, hang and mid-stream drop: it stays open for up to 10 s after the endpoint recovers and refuses requests that would have succeeded. It reduces remote load (0.85–0.89 calls per request against 2.25–2.28), which is its job, but with nowhere to send the refused requests the user sees only failures.
4. **The breaker can lose to plain retries.** Under flapping, P1 reaches 98.4% and P4 96.8%, because each up-phase lets a retry through and the breaker's open window sends traffic to an edge that cannot take the 10% long prompts. A breaker tuned for a hard outage (`threshold = 3`, `open_for = 10 s`) is mis-tuned for a flapping one; the choice depends on which failure you expect.
5. **A brownout never opens the breaker.** Successes reset the consecutive-failure count, so P3–P5 behave like P1 and the edge is never used. If half your requests fail, a threshold on *consecutive* failures is the wrong trigger; use a failure *rate* over a window.
6. **Resume pays when work was already done.** After a mid-stream drop, P5 gains 1.6 points of success and cuts p95 from 23.7 s to 16.0 s, because the edge generates only the remaining tokens. P5 is not better anywhere else, because there is nothing to resume.

## 41.6 Resume: why replay works and where it does not

Resuming does not need any cooperation from the server. The client keeps the tokens it received, appends them to the prompt and asks the next backend to continue. This is correct for greedy decoding with the same model; with sampling, or with a *different* model (the edge fallback), the continuation will not match what the first model would have said, so the text can change style or contradict its first half. The edge fallback also cannot replay a conversation that has outgrown 2048 tokens: the simulator refuses such requests, and so must a real client.

## 41.7 Experiment 2 — retries against a recovering fleet

A fleet of identical clients offers 70 req/s to a service that can complete 100 req/s. For 30 s (t = 20 to 50) capacity falls to 30 req/s. Clients give up on an attempt after 5 s and, depending on policy, retry up to three times. The server is a single FIFO queue.

| Policy | Recovery after the dip | Peak queue | Wasted work | Logical requests completed |
|---|---|---|---|---|
| F0 no retries | 30 s | 1188 [1142–1220] | 3475 | 67.1% |
| F1 immediate retries ×3 | never | 22 837 | 10 636 | 15.6% |
| F2 + jittered backoff | never | 22 445 | 10 636 | 15.6% |
| F3 + retry budget 10% | 52 s [50–55] | 1582 | 5769 | 52.5% |
| F4 + expired-request shedding | 0 s | 622 | 610 | 86.6% |
| F5 shedding only (no retries) | 0 s | 384 | 582 | 86.5% |

"Recovery" is the time after capacity returns until a 5-second window again completes at least 90% of the offered rate (63 req/s). "Wasted work" is requests the server processed after the client had already given up.

Goodput (requests completed in time, per second) in 5-second windows, seed 0:

```text
F1  68 71 71 67 30 22  0  0  0  0  0  0  0 ...        (never recovers)
F4  68 71 71 67 30 23  4  1  2  2 99 100 83 74 63 69 70 ...
```

### Reading the result

- **Hand check of the backlog.** During the dip the queue grows by `70 − 30 = 40` requests/s for 30 s: 1200 requests. F0 peaked at 1188.
- **A backlog longer than the timeout makes completed work worthless.** At 100 req/s, 500 queued requests mean a 5 s wait. Everything behind that point is abandoned by its client before it is served, but a server that does not look at request age still serves it. Capacity is spent on answers nobody is waiting for.
- **Retries make it self-sustaining.** F1 turns each abandoned attempt into a new one: the queue reaches 22 800 and the service completes nothing for the rest of the run, even though capacity returned at t = 50. This is a *metastable failure*: the trigger (the dip) has ended, but the overload keeps itself alive.
- **Jitter is not the fix.** F1 and F2 are identical in outcome. Jitter spreads retries in time; it does not reduce how many there are, and with the queue already past the timeout every spread-out retry still arrives to a hopeless queue.
- **A retry budget limits the damage.** F3 recovers after 52 s, with 52.5% of requests completed, because retries add at most 10% to the offered load and the queue can drain.
- **Shedding is the decisive change.** F4 and F5 recover in the first window after capacity returns (goodput 99 req/s, the queued backlog being served), with 86.5% of requests completed. Dropping a request whose age already exceeds the client timeout costs the server almost nothing and removes exactly the work that was wasted. F5, with no retries at all, does as well as F4: **the server-side rule did the work, and the client-side retries added nothing in this scenario.**
- **The shed rule must use information the server has.** The simulator sheds on request age against the timeout; in practice this needs a deadline or timestamp propagated in the request, or the server to cancel work when the client's connection closes.

## 41.8 Edge feasibility arithmetic

Decode on a single stream is bound by memory bandwidth: every generated token reads all the weights once. An upper bound is `bandwidth ÷ weight bytes`.

| Device (assumed bandwidth) | Model | Weights | Decode bound |
|---|---|---|---|
| phone, 60 GB/s | 3B, 4-bit | 1.5 GB | 40 tok/s |
| phone, 60 GB/s | 8B, 4-bit | 4.0 GB | 15 tok/s |
| laptop, 100 GB/s | 3B, 4-bit | 1.5 GB | 66.7 tok/s |
| laptop, 100 GB/s | 8B, 4-bit | 4.0 GB | 25 tok/s |
| 200 GB/s | 8B, 4-bit | 4.0 GB | 50 tok/s |
| 200 GB/s | 8B, 8-bit | 8.0 GB | 25 tok/s |

Hand check: 8 × 10⁹ parameters × 0.5 bytes = 4 GB; 200 ÷ 4 = 50. These are ceilings: real engines reach a fraction of them, and the simulator's 25 tokens/s edge decode corresponds to the 8B 4-bit laptop row only if the engine reaches its bound. **The bandwidth figures are assumed round numbers, not specifications of any device; look up your target's.**

### WebGPU buffer limits

The WebGPU defaults are a `maxStorageBufferBindingSize` of 134 217 728 bytes (128 MiB) and a `maxBufferSize` of 268 435 456 bytes (256 MiB). Limits higher than the default exist on many adapters but must be **requested when the device is created**; otherwise you get the defaults. A 1.5 GB model therefore needs at least 12 separate 128 MiB bindings unless the engine requests a larger limit (`ceil(1.5·10⁹ / 134 217 728) = 12`), and 4 GB needs 30. This is a design constraint on the engine, not something to work around in an application; WebLLM handles it for you, but a model that exceeds what the adapter can provide fails to load rather than running slowly.

## 41.9 The engines

**WebLLM** (`@mlc-ai/web-llm`) runs models in the browser on WebGPU and exposes an OpenAI-style API (streaming, JSON mode, function calling). It is created with `CreateMLCEngine(model, config)`. The weights are downloaded on first use and cached in the browser; the cache backend is chosen through `AppConfig.cacheBackend` (`"cache"` by default, also `"indexeddb"`, `"opfs"` and `"cross-origin"`, the last needing a browser extension). Budget the first download in your loading UI: it is a multi-hundred-megabyte transfer.

**MLX LM** (`pip install mlx-lm`) runs on Apple silicon through MLX. The command-line tools are `mlx_lm.generate`, `mlx_lm.chat` and `mlx_lm.server`; `mlx_lm.convert` with the quantize option makes a 4-bit copy of a Hugging Face model. `--prefill-step-size` (default 2048) trades peak memory during prompt processing against prompt speed. Check each command's `-h` for the options of your installed version, because they change between releases.

Neither engine is a server replacement: one stream at a time on one device, with the context limit of the model you could fit.

## 41.10 Runbook (UNTESTED)

None of the steps below has been run. Treat the first two as capability probes and the rest as a template.

**1. Does the browser have WebGPU, and with what limits?** In the page's console:

```javascript
const adapter = await navigator.gpu?.requestAdapter();
console.log(!!adapter, adapter && adapter.limits.maxStorageBufferBindingSize,
            adapter && adapter.limits.maxBufferSize);
```

Expect 134 217 728 and 268 435 456 *or higher* (the adapter's maximum); the device you get from `requestDevice()` carries only the defaults unless you pass `requiredLimits`.

**2. A minimal WebLLM call (substitute a model id from the engine's current list):**

```javascript
import { CreateMLCEngine } from "@mlc-ai/web-llm";
const engine = await CreateMLCEngine(MODEL_ID, {
  initProgressCallback: (p) => console.log(p.text),
});
const r = await engine.chat.completions.create({
  messages: [{ role: "user", content: "Say hello in five words." }],
  max_tokens: 32,
});
console.log(r.choices[0].message.content);
```

**3. Apple silicon:**

```bash
python3 -m pip install mlx-lm
mlx_lm.generate --prompt "How tall is Mt Everest?"
mlx_lm.generate -h        # confirm the options of your version
```

**4. Wiring the fallback.** Per request: (a) check the breaker; (b) try the remote call with a 3 s first-token timeout and a 30 s total deadline (`AbortController`); (c) on failure, spend a retry-budget token and wait a jittered delay, or go to the edge; (d) on a mid-stream failure, keep the received text and replay it as a prefix; (e) show the user that the answer came from a smaller local model. Log every fallback with its reason.

**5. What to measure on the device before trusting any number here:** time to first token and tokens/s for your model at your prompt lengths; peak memory; behaviour when the tab is backgrounded; battery and temperature over ten minutes of use; the first-load download time on your slowest target network.

## 41.11 Failure catalogue

| Failure | Cause | Defence |
|---|---|---|
| Retry storm after a dip | Abandoned requests are still served; retries add load | Server sheds requests older than the client timeout; retry budget; deadline propagation |
| Breaker closed by a stale success | A call issued before the breaker opened returns late | Generation counter; ignore stale results |
| Breaker stuck half-open | The probe call was cancelled or lost | Probe timeout |
| Breaker never opens on a brownout | Successes reset a consecutive-failure counter | Trip on failure rate over a window |
| Breaker adds outage after recovery | Open window outlives the failure and nothing receives the traffic | Short `open_for`, a fallback, or a lower threshold for probing |
| Breaker flaps against a flapping endpoint | Window shorter than the flap period | Match `open_for` to the failure shape; consider plain retries there |
| Immediate retries on a fast failure | All attempts fail in the same second | Backoff; do not retry connection-refused beyond one attempt |
| Backoff without jitter | Clients that failed together return together | Full jitter |
| Retried non-idempotent request | Replay duplicates a side effect | Idempotency key; do not retry tool-calling requests blindly |
| Mid-stream restart pays twice | Received tokens discarded | Keep tokens and replay as prefix (greedy/same model only) |
| Edge answer differs from remote | Smaller or different model, or sampling | Label the answer; avoid resume across models when consistency matters |
| Edge cannot hold the request | Prompt plus output exceeds the edge context | Check length before routing to the edge; fail fast with a clear message |
| Edge overload | One request at a time; queue grows | Bounded queue; reject the third waiting request |
| Weights exceed buffer limits | WebGPU default 128 MiB binding / 256 MiB buffer | Request higher limits at device creation; choose a model the adapter can hold |
| First edge use is a long download | Weights fetched on first load | Prefetch when idle; show progress; cache backend |
| Fallback never exercised | Only used in outages | Inject faults on a schedule; run the fallback path in tests |

## 41.12 Exercises

1. In Experiment 1 under *refused*, P1 equals P0 in success. Explain why in one sentence, and say what change to the retry policy would make retries useful against a fast failure.
2. P3 scores 55.8% against P0's 57.3% under *hang*. Is the breaker broken? What would you add so the breaker helps rather than hurts?
3. Compute the number of remote calls per request expected from P1 under *hang* if the outage covered 25% of the timeline, with four attempts during the outage and one otherwise.
4. In the fleet model, the offered load is 70 req/s, capacity is 100 req/s, the dip is 30 s at 30% capacity and the client timeout is 5 s. Estimate the queue length at the end of the dip and how long, with no retries and no shedding, the service needs to drain it to the point where a new request is again served within the timeout.
5. A 7B model in 4-bit uses 3.5 GB of weights. On a device with 120 GB/s of memory bandwidth, what is the decode ceiling, and how many 128 MiB storage bindings does the model need?

### Solutions

1. All four attempts happen within about a second, and the outage lasts two minutes, so every retry lands inside it. Retries are only useful against a fast failure if they are spread over the outage's duration (a long backoff, or a periodic probe by the breaker) or sent to a different backend.
2. It is working as designed: it opened and then refused requests for up to `open_for` seconds after recovery, with nowhere to send them. Add a fallback (the edge, or a degraded response), or keep `open_for` short so the probe finds the recovery quickly.
3. `0.25 × 4 + 0.75 × 1 = 1.75` calls per request.
4. Growth: `(70 − 30) × 30 = 1200` requests. A new request is served within 5 s when the queue holds fewer than `100 × 5 = 500` requests ahead of it. After the dip the queue shrinks at `100 − 70 = 30` req/s, so it takes `(1200 − 500) / 30 ≈ 23 s`. Experiment 2 measured 30 s for F0, which includes the 5 s window granularity and the 90% threshold.
5. Decode ceiling `120 / 3.5 ≈ 34.3 tok/s`. Bindings: `3.5·10⁹ / 134 217 728 = 26.07`, so 27.

## 41.13 Summary

- A remote endpoint fails in five shapes; each defeats a different policy. Design against the shapes you expect, and test against all five.
- Retries do nothing for a fast failure, and they multiply the load during a slow one (2.2× remote calls here).
- A circuit breaker saves remote load (0.85–0.89 calls per request against 2.25–2.28) but, without a fallback, costs availability after recovery. It can also lose to plain retries against a flapping endpoint, and it never opens on a brownout if it counts consecutive failures.
- An edge fallback bounds the outage at what the edge cannot serve: 40% down became about 4% failed, with a latency cost (p95 20–24 s) and a quality cost the simulator does not measure.
- Resuming after a mid-stream drop cut p95 from 23.7 s to 16.0 s and raised success by 1.6 points, for the one failure where work was already done.
- A retry storm is a server problem as much as a client one: dropping requests whose clients have already given up restored goodput at once; jitter alone did not, and neither did retries.
- Edge capacity is set by memory bandwidth and by what fits the WebGPU limits; measure both on your target devices.
- Everything here is simulated. The WebGPU, WebLLM and MLX steps have not been run.

## Files and sources

- Code: `docs/examples/inference-systems/ch41_edge_fallback.py`
- WebGPU limits (`GPUSupportedLimits`): <https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits>
- WebLLM repository and README (OpenAI-style API, `CreateMLCEngine`, cache backends): <https://github.com/mlc-ai/web-llm>
- WebLLM paper: <https://arxiv.org/abs/2412.15803>
- MLX LM repository and README: <https://github.com/ml-explore/mlx-lm>
- MLX LM on PyPI (command-line options such as `--prefill-step-size`): <https://pypi.org/project/mlx-lm/>
