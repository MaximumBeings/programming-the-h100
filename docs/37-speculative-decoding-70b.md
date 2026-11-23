# Chapter 37: Speculative Decoding at 70B: Multi-GPU Target, Cost Model and Measured Speedup

!!! abstract "Goal and prerequisites"

    Predict, before renting any GPUs, whether speculative decoding will speed up a 70B model on 2 x H100, which draft and which depth K to use, and at what batch size the benefit disappears. Then measure it with the Chapter 36 harness. Read [Chapter 32](32-inference-engine-and-harness.md) (decode floor) and [Chapter 36](36-benchmark-teardown.md) (measurement). The statistics of rejection sampling are derived in the Triton book's speculative decoding chapter; here the focus is the multi-GPU system cost.

!!! warning "Evidence level: read this first"

    - **Verified here (CPU):** the cost model, the expected-tokens formula (against Monte Carlo), eight property tests, and the tables below.
    - **Assumed:** `mfu = 0.5`, a 1 ms fixed per-step overhead, and the acceptance rates alpha in the tables. Alpha comes from the draft and the data, and **no acceptance rate in this chapter was measured on a real model pair**. Every speedup figure is a model prediction, not a benchmark.
    - **Untested on H100:** the launch commands in Section 37.8. Flag names change between releases; check `--help` for your version.

## 37.1 Why speculation helps, and exactly when

Decode of a large model is limited by reading weights from HBM, not by arithmetic. For 70B parameters in BF16 on 2 GPUs (tensor parallel 2), each GPU holds 70 GB and one step takes at least `70e9 / 3.35e12 = 20.9 ms`. The step reads the same weights whether it processes one token or a hundred, until the arithmetic catches up with the memory traffic.

Speculative decoding exploits that slack. A small **draft** proposes K tokens cheaply; the large **target** checks all K (plus one more position) in a **single** pass. Rejection sampling accepts a prefix of the proposals and always yields one extra token, and the output distribution is exactly the target's. If the pass costs little more than a one-token pass, several tokens are emitted for roughly the price of one.

The benefit depends on three quantities:

1. **alpha**, the per-token acceptance probability (how well the draft imitates the target).
2. **The draft's cost** relative to the target step.
3. **How much slack the target step has**: the verify pass processes `batch x (K+1)` tokens, and once that count exceeds the compute-bound threshold, verifying extra tokens is no longer free.

The vLLM documentation makes the same qualitative point: speculation reduces inter-token latency for memory-bound workloads at low to medium request rates, which is why it is not a free win at high load.

## 37.2 The cost model

```python
def step_time(m, tokens, mfu=0.5, overhead=1.0e-3, comm=0.0):
    mem  = m.weight_bytes_per_gpu() / HBM_BW
    comp = 2.0 * m.params * tokens / (m.gpus * PEAK_BF16 * mfu)
    return max(mem, comp) + overhead + comm
```

One forward over `tokens` total tokens costs the larger of the weight-read time and the arithmetic time, plus a fixed overhead (kernel launches, sampling, scheduler) and communication. This is a roofline model: coarse, but it captures the crossover that decides everything.

Expected tokens per verify pass with per-token acceptance alpha and K proposals:

```
E[tokens] = (1 - alpha^(K+1)) / (1 - alpha)
```

One speculation cycle costs K sequential draft steps plus one verify pass of `(K+1)` tokens per sequence:

```python
def cycle_time(target, draft, K, batch):
    return K * step_time(draft, batch) + step_time(target, batch * (K + 1))

speedup = expected_tokens(alpha, K) * step_time(target, batch) / cycle_time(...)
```

### Where the compute threshold lies

With 70B parameters on 2 GPUs, one token costs `2 x 70e9 = 140 GFLOP`. At an assumed 50 percent of the 989 TFLOP/s peak per GPU, the arithmetic time equals the 20.9 ms memory time near 295 tokens in flight. The model's base step times confirm the shape:

| Batch | Base step (ms) |
|---|---|
| 1 to 128 | 21.9 |
| 256 | 37.2 |
| 512 | 73.5 |

Up to about 128 sequences the step is memory-bound (21.9 ms is the 20.9 ms floor plus 1 ms overhead). Beyond that it turns compute-bound and grows with batch.

## 37.3 Verified properties

The file `ch37_spec_decode_70b.py` ([download](examples/inference-systems/ch37_spec_decode_70b.py)) checks the model against eight properties:

```
  [PASS] 70B TP2 decode floor ~20.9 ms (20.90 ms)
  [PASS] E[tokens] formula matches Monte Carlo (max err 0.0052)
  [PASS] limits: alpha=0 gives 1, alpha=1 gives K+1
  [PASS] speedup increases with alpha
  [PASS] best K nondecreasing in alpha ([1, 1, 2, 5])
  [PASS] speedup nonincreasing in batch ([1.64, 1.64, 1.57, 0.64, 0.57])
  [PASS] acceptance(p,p)=1, disjoint=0
  [PASS] draft as large as target gives speedup < 1
```

The Monte Carlo run simulates the accept-or-reject chain directly (200,000 cycles for each of 16 alpha and K pairs), so the closed form is checked against the process it describes. The last test is the cautionary one: a draft as expensive as the target loses to plain decoding even with a good acceptance rate.

## 37.4 Predicted speedup tables (assumptions, not measurements)

Best K searched over 1 to 12 at each cell. `mfu = 0.5`, overhead 1 ms, alpha as shown.

**Draft: 8B BF16 on one GPU**

| alpha | B=1 | B=8 | B=32 | B=128 | B=512 |
|---|---|---|---|---|---|
| 0.5 | 1.19x (K=1) | 1.19x (K=1) | 1.19x (K=1) | 0.76x (K=1) | 0.67x (K=1) |
| 0.7 | 1.43x (K=2) | 1.43x (K=2) | 1.43x (K=2) | 0.87x (K=1) | 0.76x (K=1) |
| 0.8 | 1.65x (K=3) | 1.65x (K=3) | 1.65x (K=3) | 0.92x (K=1) | 0.81x (K=1) |
| 0.9 | 2.02x (K=5) | 2.02x (K=5) | 1.92x (K=3) | 0.97x (K=1) | 0.85x (K=1) |

**Draft: 1B BF16 on one GPU**

| alpha | B=1 | B=8 | B=32 | B=128 | B=512 |
|---|---|---|---|---|---|
| 0.5 | 1.54x (K=3) | 1.54x (K=3) | 1.54x (K=3) | 0.85x (K=1) | 0.74x (K=1) |
| 0.7 | 2.16x (K=5) | 2.16x (K=5) | 2.08x (K=3) | 0.96x (K=1) | 0.84x (K=1) |
| 0.8 | 2.75x (K=7) | 2.75x (K=7) | 2.45x (K=4) | 1.01x (K=1) | 0.89x (K=1) |
| 0.9 | 3.98x (K=11) | 3.98x (K=11) | 2.99x (K=4) | 1.07x (K=1) | 0.94x (K=1) |

What the tables say, and what they do not:

1. **Low and medium batch: real gains.** Up to B=32 the model predicts 1.2x to 4.0x depending on alpha and the draft. Gains grow with alpha and with a cheaper draft.
2. **Large batch: speculation loses.** At B=128 and above almost every cell is below 1.0, because the verify pass of `batch x (K+1)` tokens is already compute-bound and the extra proposals are rejected work. This is why a serving system that runs at high batch should switch speculation off, or pick K per load level.
3. **The best K moves with alpha and with the draft.** The 8B draft at alpha 0.9 wants K=5; the 1B draft wants K=11. The K=11 optimum sits on a very flat curve (K=10 to 12 differ by under 0.5 percent), so treat it as "deep", not as a precise optimum, and note that real alpha usually **decays with position**, which this i.i.d. model ignores (Section 37.6).
4. **Everything depends on alpha, which is not in this chapter's data.** The tables show sensitivity; they do not tell you which row applies to your model pair and prompts. Measure it (Section 37.7).

### Worked Example 37.4.1: one cell by hand

70B target, 1B draft, alpha 0.8, K=7, batch 1.

- Expected tokens: `(1 - 0.8^8) / 0.2 = (1 - 0.1678) / 0.2 = 4.16`.
- Draft step: weights `1e9 x 2 = 2 GB` on one GPU, `2e9 / 3.35e12 = 0.60 ms`, plus 1 ms overhead = 1.60 ms. Seven of them: 11.2 ms.
- Verify: 8 tokens in flight, memory-bound, 21.9 ms.
- Cycle: 33.1 ms. Base step: 21.9 ms.
- Speedup: `4.16 x 21.9 / 33.1 = 2.75`. This matches the table.

Notice that the **fixed overhead dominates the draft cost**: 1 ms of overhead is larger than the 0.6 ms of weight reading. Cutting launch and scheduling overhead (CUDA graphs, fused sampling) matters more for a tiny draft than shrinking it further.

## 37.5 Multi-GPU placement of the draft

The target occupies 2 GPUs with tensor parallelism. The draft must run somewhere, and the choice changes the cost:

| Placement | Draft step cost | Costs | When it fits |
|---|---|---|---|
| Replicated on each target GPU, same TP | Draft weights are sharded and an all-reduce per layer is added; for a small model the communication can exceed the arithmetic | Little extra memory | Large draft (8B and up) |
| Draft on one target GPU, no TP (`draft_tensor_parallel_size = 1` in vLLM) | One GPU computes while the other idles; the proposals must reach the other rank | Idle GPU during drafting | Small drafts such as 1B |
| Draft on a separate GPU | No contention with target memory | One more GPU | Memory-limited targets |
| Draft heads on the target (EAGLE, MTP, Medusa style) | Very cheap: reuse target hidden states | Training or a published head is required | When one exists for your model |

The model in this chapter treats the draft as a single-GPU model (the 8B and 1B rows). It does not model the communication of sharing proposals between ranks; add a `comm` term from your interconnect measurements.

Memory check for the target at TP=2: weights 70 GB per GPU leave about 10 GB per GPU for KV cache at an 80 GB card, before activations and the draft. A 1B BF16 draft is 2 GB; an 8B draft is 16 GB, which may not fit beside the target on the same GPU at long contexts. Check this before choosing a draft.

## 37.6 Failure modes and edge cases

| Situation | Effect | Mitigation |
|---|---|---|
| alpha drops with position (later proposals diverge more) | Real expected tokens are below the i.i.d. formula; best K is smaller | Measure acceptance per position and use `E = 1 + sum_i prod_{j<=i} a_j` |
| High temperature or creative text | Draft and target distributions overlap less | Tune K per workload; consider n-gram or suffix methods for repetitive content |
| Batch grows at peak traffic | Verify turns compute-bound and speculation slows everything | Dynamic speculation depth, or disable above a batch threshold |
| Draft and target tokenizers differ | Proposals cannot be verified | Use a draft from the same family and tokenizer |
| Draft too large or too slow | Cycle cost exceeds the savings (Test 8) | Compare draft step time with target step time before anything else |
| Different sampling params between phases | Violates the rejection-sampling guarantee | Apply identical temperature, top-p and top-k to both distributions |
| Prefix caching or other features interact | Support varies by engine and release; the SGLang docs list limitations for some modes | Check the release notes for your version |
| Acceptance measured on one dataset, served on another | Surprise slowdown | Measure on the production distribution |

### The synthetic alpha experiment, and why not to over-read it

The file also computes alpha from distributions: `alpha = sum_x min(p(x), q(x))`, the overlap of target `p` and draft `q`. With draft logits equal to target logits plus Gaussian noise of standard deviation 0.8, over 400 random 50-token distributions:

```
  T=0.2 : alpha=0.708
  T=0.5 : alpha=0.726
  T=1.0 : alpha=0.782
  T=1.5 : alpha=0.818
```

Alpha **rises** with temperature here because a higher temperature flattens both distributions and shrinks the effect of the logit noise. That is a property of this synthetic noise model, not a statement about real models; with real drafts that disagree on the preferred token, the trend can go the other way. The experiment's value is that it shows the correct definition of alpha and a way to compute it from logged distributions. Do not quote its numbers.

## 37.7 Measuring it: protocol on real hardware

Use [Chapter 36](36-benchmark-teardown.md)'s open-loop client and manifest. Add the speculation-specific steps:

1. **Baseline first.** Run the target alone, no speculation, same engine, same flags, at the same offered rates.
2. **Measure alpha per position.** Most engines expose acceptance or accepted-length metrics (SGLang exposes `sglang:spec_*` metrics; vLLM logs speculative decoding metrics). Record mean accepted length and per-position acceptance on **your** prompts.
3. **Sweep K** (for example 1, 2, 3, 5, 8) at low and high offered load. Plot speedup against K for each load.
4. **Sweep load** to find the batch size at which speculation stops helping on your hardware. Compare it with the model's prediction near batch 128; a different crossover tells you the real MFU and overhead.
5. **Check output equivalence.** At temperature 0, outputs with and without speculation should match token for token on the vast majority of prompts; small differences can come from numerical non-determinism in batched kernels. At temperature above 0, compare distributions statistically (for example the total variation test in the Triton book's speculative chapter), not strings.
6. **Report the SLO view.** Speculation helps TPOT at low load and can hurt TTFT slightly (extra prefill for the draft) and tail latency at high load. Report p99 TTFT, p99 TPOT and goodput, not only mean speed.
7. **Calibrate the model.** Fit `mfu` and `overhead` so the model matches your measured base step times, then compare predicted and measured speedup. The gap is what your engine's real behavior adds.

### What you can compare against

The SGLang documentation publishes a single-H100 comparison on a smaller model: roughly 158 tokens per second without speculation, 244 with EAGLE-2 and 373 with EAGLE-3. These are vendor figures on one model and one GPU, useful as an order of magnitude (1.5x to 2.4x with trained draft heads), and say nothing about 70B on two GPUs.

## 37.8 Runbook (untested on H100)

!!! warning "UNTESTED ON H100"

    Commands follow the vLLM and SGLang documentation for the versions visible when this chapter was written. Verify every flag with `--help` for your exact version.

```bash
# --- vLLM: separate draft model, target on 2 GPUs, draft on 1 ---
vllm serve <70B-model> --tensor-parallel-size 2 \
  --speculative-config '{"method":"draft_model","model":"<small-draft-model>",
                         "num_speculative_tokens":5,"draft_tensor_parallel_size":1}'

# --- vLLM: no draft model (lookup based), cheap to try first ---
vllm serve <70B-model> --tensor-parallel-size 2 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'

# --- SGLang: EAGLE-3 draft head ---
python -m sglang.launch_server --model-path <70B-model> --tp-size 2 \
  --speculative-algorithm EAGLE3 --speculative-draft-model-path <eagle3-draft> \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

Notes from the documentation: vLLM takes all speculative settings through one `--speculative-config` JSON and has deprecated the older separate flags; its `draft_tensor_parallel_size` key sets the draft's parallelism; SGLang's tree parameters (`num-steps`, `eagle-topk`, `num-draft-tokens`) interact and should be tuned together with `--cuda-graph-max-bs` and memory settings. Always run the baseline with the same flags minus the speculative ones.

## 37.9 Exercises

1. Using the formulas, compute the predicted speedup for the 8B draft at alpha 0.7, K=2, batch 1. Compare with the table.
2. For the 1B draft at alpha 0.9, find the K at which the speedup stops improving, and explain what limits it.
3. Why does the speedup at batch 512 fall below 1.0 for every alpha in the tables, even alpha 0.9?
4. The per-position acceptance is 0.9, 0.8, 0.7, 0.6, 0.5 for positions 1 to 5. Compute expected tokens per pass and compare with the i.i.d. formula using alpha equal to the mean.
5. A draft step costs 0.6 ms of weight reading plus 1 ms overhead. Which is the better engineering investment: halving the draft's weights, or halving the overhead?
6. Your measured baseline step is 25 ms, not 21.9. List two explanations and how you would tell them apart.
7. Design the experiment that decides whether to turn speculation off above a batch threshold.

## 37.10 Solutions

1. Expected tokens `(1 - 0.7^3)/0.3 = 2.19`. Draft: 8B on one GPU, `16e9/3.35e12 = 4.78 ms` plus 1 ms = 5.78 ms; two steps 11.6 ms. Verify 21.9 ms. Cycle 33.5 ms. Speedup `2.19 x 21.9 / 33.5 = 1.43`, matching the table.
2. The curve peaks at K=11 and is flat around it: 3.87x at K=8, 3.97x at K=10, 3.98x at K=11, 3.98x at K=12, then falling (3.85x at K=16, 3.62x at K=20). Each extra draft step costs a fixed 1.6 ms, while the extra expected tokens shrink as `0.9^K`, so the gain per step eventually falls below the cost. In real models alpha decays with position and the true optimum is smaller.
3. At 512 sequences the verify pass processes `512 x (K+1)` tokens, deep in the compute-bound region where cost grows linearly with tokens. Proposals that are rejected cost real compute, and the base step itself is already 73.5 ms with no free slack left.
4. Expected tokens `= 1 + 0.9 + 0.9x0.8 + 0.9x0.8x0.7 + ... = 1 + 0.9 + 0.72 + 0.504 + 0.3024 + 0.1512 = 3.58`. The mean alpha is 0.7; the i.i.d. formula with K=5 gives `(1 - 0.7^6)/0.3 = 2.94`. The i.i.d. formula underestimates here because the early positions are higher than the mean; the converse holds when early acceptance is lower. Always use measured per-position values.
5. The overhead. Weight reading is 0.6 ms and overhead is 1 ms, so halving overhead saves 0.5 ms per draft step against 0.3 ms for halving the weights, and the overhead also applies to larger drafts. Both matter less than raising alpha.
6. Candidate causes: lower achievable memory bandwidth than the 3.35 TB/s peak (real kernels reach perhaps 70 to 90 percent), or larger overhead and communication (tensor-parallel all-reduce). Tell them apart by measuring effective bandwidth of a pure weight-read kernel, and by profiling the step for all-reduce time.
7. Run the baseline and speculative configurations across offered rates and record the batch actually in flight; find the rate at which speculative p99 TPOT crosses the baseline; set the threshold below it with margin; verify with a ramp test that goodput does not drop at the switch point.

## Chapter Summary

For a memory-bound 70B model on 2 H100s the decode floor is 20.9 ms per token, and a verify pass over K+1 tokens costs almost the same until about 128 sequences are in flight. Speculative decoding converts that slack into speed: expected tokens per pass is `(1 - alpha^(K+1))/(1 - alpha)`, a cycle costs K draft steps plus one verify, and the cost model predicts roughly 1.2x to 4x at low batch, depending on acceptance and draft cost, and slowdowns at high batch where the verify pass is compute-bound. The acceptance rate is the dominant unknown and must be measured per position on production prompts; the model's numbers are predictions under assumed MFU and overhead and have not been measured on hardware. Fixed overhead, not weight size, dominates a tiny draft's cost; the draft's placement across GPUs adds communication and memory constraints; and the sound deployment pattern is to measure a baseline, sweep K and load with the open-loop harness, and switch speculation off or shorten K above the batch where it stops helping.

## Sources

- vLLM speculative decoding documentation: <https://docs.vllm.ai/en/latest/features/speculative_decoding/index.html>
- vLLM draft model guide: <https://docs.vllm.ai/en/v0.17.1/features/speculative_decoding/draft_model/>
- SGLang speculative decoding documentation: <https://docs.sglang.io/docs/advanced_features/speculative_decoding>
- SGLang adaptive speculative decoding: <https://sgl-project.github.io/advanced_features/adaptive_speculative_decoding.html>
