# Chapter 46 — Capstone: A Mini Inference Engine

!!! warning "Evidence status"
    **Run on CPU:** a tiny real transformer (2 layers, 32-dimensional, 2 heads, vocabulary 96, random weights, float64) served by an engine that joins the paged KV cache (Chapter 43), the continuous-batching scheduler with preemption and chunked prefill (Chapter 44) and a block-level prefix cache (Chapter 45); 15 self-tests; four experiments (`ch46_mini_engine.py`).
    **What "correct" means here:** every generated token is compared with a *reference* that recomputes the whole sequence with dense causal attention and no cache at every step. In float64 with greedy decoding the comparison is exact.
    **What this does not show:** nothing runs on a GPU, the weights are random so the text is meaningless, sampling is greedy only, and the engine computes one sequence at a time (the "batch" is a scheduling decision; its cost uses the assumed per-iteration model of Chapter 44). It checks that the *machinery* is correct, not that it is fast. The comparison with vLLM and SGLang in 46.9 is **UNTESTED**.

This chapter closes Part XIV. The earlier chapters each tested one piece in isolation. Here the pieces share one set of blocks, and the property to check is the one that matters in production: **sharing, preempting, chunking and caching must never change an answer.**

## 46.1 What is joined

```
 requests ──► scheduler (Ch 44) ──► per-iteration plan: decode set + prefill chunks
                 │  admit / preempt / chunk                │
                 ▼                                          ▼
        block allocator + refcounts (Ch 43) ◄── prefix cache (this chapter, block-level trie;
                 │                                   same semantics as Ch 45 rounded to blocks)
                 ▼
        KV pool [layers, blocks, B, heads, dim]  ◄── model forward (tiny transformer)
                                                         prefill chunk: gather + causal attention
                                                         decode: walk the block table, online softmax
```

| Component | Source | What the engine adds |
|---|---|---|
| Block allocator with reference counts | Chapter 43 | one allocator shared by sequences *and* the cache |
| Block tables, paged attention | Chapter 43 | real K/V for every layer, written and read through the tables |
| Admission, preemption, watermark, chunked prefill | Chapter 44 | the same decisions, now driving real computation |
| Prefix reuse | Chapter 45 | exact block-level trie holding references to physical blocks |

## 46.2 The model and the reference

`TinyLM` is a decoder-only transformer: token embedding plus fixed sinusoidal positions, then per layer an RMS norm, multi-head causal attention and a ReLU feed-forward block with residual connections, and a final norm with logits through the transposed embedding matrix. The weights are random. That is deliberate: the aim is to test the machinery, and a random model is as sensitive to a misplaced key or a wrong position as a trained one (test 13 shows it).

The **reference** is `ref_generate`: at every step it runs `forward_full` over the entire sequence (prompt plus the tokens so far) with a dense causal mask and takes the argmax of the last position. It uses no cache, no blocks and no batching, so it cannot share their bugs.

## 46.3 Running the model over paged memory

`forward_chunk(tokens, start, block table)` runs tokens at positions `start …` and does the following per layer:

1. compute `q, k, v` for the chunk;
2. write `k, v` into the pool at `table[pos // B]`, slot `pos % B`;
3. attend:
   - **a prefill chunk (several tokens)** gathers the sequence's K/V through the table and applies dense attention with a causal mask offset by `start`;
   - **a decode step (one token)** walks the block table with an online softmax, never gathering.

Both paths exist in real engines (different kernels for prefill and decode). Test 2 checks them against the full forward pass for `B = 1, 4, 8, 16` using mixed chunk sizes (5, 1, 1, 12, 1, 17 tokens), and test 3 checks the block walk against gathering on a table whose blocks are scattered.

## 46.4 The engine loop

The loop is Chapter 44's, with the simulated token counts replaced by real computation. The points where the pieces interact:

- **Admission.** For a waiting sequence the engine takes `tokens = prompt + generated so far`, asks the prefix cache for the longest run of cached whole blocks, and **pins** them with `incref` before deciding anything, so they cannot be evicted while the decision is made. It then needs `⌈target / B⌉ − matched` fresh blocks, evicting cache entries if necessary. If they cannot be found, the pins are undone and the head of the queue waits (strict first-come-first-served).
- **At least one token is always computed.** The match is capped at `(len − 1) // B` blocks. If the whole prompt were cached there would be nothing to run and no logits to pick the next token from.
- **Shared blocks are read-only.** Only *full* blocks are shared, and a sequence's own writes always land in a later block, so no copy-on-write is needed here. (Sharing a partial block would need Chapter 43's copy-on-write; Exercise 6.)
- **Registration.** When a sequence finishes its prefill, and again when it finishes generating, its full blocks are added to the trie; the cache takes one reference to each block it adds. Two sequences that compute the same prefix at the same time each keep their own blocks; the second one's blocks are simply not registered.
- **Eviction.** A cache entry may be evicted only if it is a leaf of the trie and the cache holds the only reference to its block. Evicted blocks go back to the allocator. The engine evicts when it needs a block for admission or for a decoding sequence, before it considers preempting anyone.
- **Preemption.** As in Chapter 44, the youngest running sequence loses its blocks and goes to the front of the queue with `target = prompt + tokens generated`. If the cache is on, a preempted sequence that had already finished its prefill finds its own prompt blocks in the cache when it is readmitted.
- **Reference counts as the invariant.** After every iteration (with `debug=True`) the engine checks that each block's reference count equals the number of running sequences whose tables contain it plus the number of cache entries pointing to it. A leak, a double free or a missing pin shows up here.

## 46.5 Self-tests (15, all pass)

| # | What is checked |
|---|---|
| 1 | the reference model is causal: changing later tokens leaves earlier logits bit-identical |
| 2 | paged chunked forward equals the full forward for B = 1, 4, 8, 16 with mixed chunk sizes (max abs diff 6.2e-15) |
| 3 | decode attention walking scattered blocks equals gathered attention |
| 4 | one request equals the reference for B ∈ {2, 5, 8, 16} and chunk budgets (none, 3, 16) |
| 5 | 24 concurrent requests, roomy pool: every output equals the reference |
| 6 | tiny pool forcing preemption and recomputation: outputs still equal the reference |
| 7 | chunked prefill with a 12-token budget: outputs equal the reference |
| 8 | prefix cache: outputs equal the reference, 600 prompt tokens served from cache, and `prefill tokens + cache hits = prefill tokens without the cache` exactly |
| 9 | cache + memory pressure + chunking in three configurations (B = 8, 4, 16): outputs equal the reference and reference counts are exact after every iteration |
| 10 | after the run and a full cache flush, every block is back on the free list |
| 11 | **with the cache off, the engine's first-token times, finish times, iteration counts and preemption counts equal the Chapter 44 simulator exactly**, request by request, in four configurations |
| 12 | the engine's block trie equals the Chapter 45 radix tree's match rounded down to whole blocks (10 sessions × 40 sequences) |
| 13 | a check of the test itself: corrupting the cached K/V blocks makes later outputs disagree with the reference |
| 14 | a request larger than the whole pool is rejected; the other still equals the reference |
| 15 | compressing the arrival times (a different batch composition) changes no output |

Test 11 matters more than it looks. It shows that the Chapter 44 simulator, whose numbers filled five tables, makes the same scheduling decisions as a real engine implementation. It does not validate the assumed cost constants; those still need hardware.

## 46.6 Experiment 1 — correctness matrix

48 requests; 60% start with one of three shared 40-token prompts. Prompt tokens in total: 2925; output tokens: 546. Every row is compared with the no-cache reference.

| configuration | mismatches | iterations | preemptions | prefill tokens | cache-hit tokens | simulated end (s) |
|---|---|---|---|---|---|---|
| B=8, roomy pool, no cache | **0** | 97 | 0 | 2925 | 0 | 1.71 |
| B=8, tiny pool (26 blocks), no cache | **0** | 228 | 4 | 3224 | 0 | 3.69 |
| B=8, chunked prefill (budget 12) | **0** | 299 | 0 | 2925 | 0 | 4.74 |
| B=8, roomy, prefix cache | **0** | 101 | 0 | 1685 | 1240 | 1.71 |
| B=8, tiny pool, prefix cache | **0** | 211 | 7 | 2338 | 1104 | 3.39 |
| B=4, 60 blocks, budget 10, prefix cache | **0** | 317 | 5 | 2276 | 940 | 4.98 |
| B=16, 14 blocks, budget 20, prefix cache | **0** | 269 | 6 | 2486 | 880 | 4.27 |

All seven give the same answers as the reference. **Hand check:** with the cache on and a roomy pool, `1685 + 1240 = 2925`: the cache removes exactly the tokens it served. The 1240 hits are 31 requests × 40 tokens (the shared prompt is 5 blocks of 8).

## 46.7 Experiment 2 — what the prefix cache buys

| configuration | prefill tokens | preemptions | TTFT p50 (ms) | TTFT p99 (ms) | e2e p50 (s) |
|---|---|---|---|---|---|
| roomy, no cache | 2925 | 0 | 31.6 | 46.6 | 0.220 |
| roomy, cache | 1685 | 0 | 30.3 | 38.7 | 0.208 |
| tiny pool, no cache | 3224 | 4 | 1136 | 2070 | 1.295 |
| tiny pool, cache | 2338 | 7 | 991 | 1770 | 1.144 |

- With room to spare the cache removes 42% of the prefill tokens but moves the latency little (TTFT p50 31.6 → 30.3 ms): at this load the system is not saturated and, in the assumed cost model, an iteration has a large fixed cost.
- **With a tiny pool, the cache increased preemptions (4 → 7) and still helped.** It holds blocks that running sequences could otherwise use, so running sequences are preempted more often. But prefill work fell from 3224 to 2338 tokens (−27%), because a preempted sequence that was readmitted found its own prompt blocks in the cache instead of recomputing them, and TTFT p50 fell from 1136 to 991 ms. The preemption count alone would have suggested the opposite conclusion.

## 46.8 Experiments 3 and 4

**Block size and hits.** The shared prompt is 40 tokens; roomy pool.

| B | cache-hit tokens | prefill tokens | `⌊40/B⌋ × B` |
|---|---|---|---|
| 4 | 1240 | 1685 | 40 |
| 8 | 1240 | 1685 | 40 |
| 16 | 992 | 1933 | 32 |
| 32 | 992 | 1933 | 32 |

Each hit saves `⌊40/B⌋ × B` tokens, 40, 40, 32 and 32. This is the alignment loss of Chapter 45 on real data: 8 tokens per hit at `B = 16`, and 31 hits × 8 = 248 more prefill tokens, as the table shows (1933 − 1685).

**Mutation check.** After the first iteration in which the cache is non-empty, the script corrupts every cached K/V block (K × −3, V + 5). Output differs from the reference for **11 of 48 requests**. A test that always passed would prove nothing; this one fails when a block is wrong.

## 46.9 Failure modes and edge cases

| Situation | Handled by |
|---|---|
| Wrong position for a recomputed sequence | outputs differ from the reference (tests 6, 9) |
| Whole prompt cached, nothing left to compute | match capped at `(len − 1) // B` blocks |
| Cache evicts a block a sequence is about to use | matched blocks pinned with `incref` before the admission decision |
| Cache entry evicted while a descendant exists | only trie leaves are evicted |
| Leaked or double-freed block | per-iteration reference-count audit; flush test (10) |
| Two sequences compute the same prefix concurrently | both keep private blocks; only the first is registered |
| Request that can never fit | rejected on arrival (test 14) |
| Cache holding blocks starves running sequences | evict before preempting; Experiment 2 shows the cost |

**Not modelled:**
- **Batched computation.** Sequences are computed one at a time in Python; a real engine batches the matrix multiplies of all decoding sequences into one kernel.
- **Numerics.** Float64 and greedy decoding make exact equality testable. In BF16 on a GPU, batched kernels round differently from sequential ones, and a near-tie can flip an argmax. A GPU version of these tests would compare log-probabilities within a tolerance and report the rate of token agreement (Exercise 7).
- **Sampling, stop tokens, speculative decoding, LoRA adapters, multi-GPU parallelism, quantised KV.**
- **Quality.** The weights are random.

## 46.10 Comparing with vLLM and SGLang on a real GPU (UNTESTED)

Command-line options are from memory; check them with `--help` for the versions you install.

1. **Pick a small model** that fits one GPU, and fix a set of prompts (use the Chapter 36 client's prompt files) with a shared system prompt for half of them.
2. **Start each engine on the same model and GPU, one at a time.**
   `vllm serve MODEL --port 8000 --enable-prefix-caching --max-num-seqs 128 --max-num-batched-tokens 2048`
   `python -m sglang.launch_server --model-path MODEL --port 8001`
3. **Agreement check** at temperature 0: send each prompt to both engines and to the reference implementation in `transformers` (single sequence, no batching). Record the fraction of prompts whose first 64 tokens agree, and the maximum log-probability difference at the first divergence. Expect a few divergences from rounding, not from sharing logic; a large number of them points at a bug or a configuration difference (a different chat template or a different default for sampling).
4. **Performance:** run the open-loop client at 2, 5, 8 and 12 req/s against each engine; record TTFT, TPOT, throughput and, from the engines' own metrics, the prefix-cache hit rate and the preemption count. Compare the shape of the curves with Experiments 1 and 5 of Chapter 44 and Experiment 1 of Chapter 45. The absolute numbers are not comparable with this chapter's.
5. **Replace the assumed constants** (`T0`, `CP`, `CD`) in the Chapter 44 simulator with values fitted from a short profile on your GPU, and check whether the simulator reproduces the engines' *relative* behaviour (the rate where TTFT turns up, the effect of the block size).

## 46.11 Exercises

1. Why is the prefix match capped at `(len − 1) // B` blocks? What happens for a 16-token prompt with `B = 8` that is entirely cached?
2. In Experiment 1, row 4, explain why `1685 + 1240 = 2925`.
3. With a tiny pool, the cache raised preemptions from 4 to 7 yet lowered prefill work by 27%. Explain both.
4. Test 11 requires exact equality of times with the Chapter 44 simulator. What does passing show, and what does it not show?
5. Predict which tests would fail if, after a preemption, `target` were set to `len(prompt)` instead of `len(prompt) + len(out)`. Then say how you would check the prediction.
6. Sharing is limited here to full blocks. Describe what would be needed to share a partially filled last block between two sequences, and what it costs.
7. Propose a tolerance-based version of test 5 for a BF16 GPU engine.

### Solutions

1. The engine must run at least one token through the model to get the logits for the next token. If a 16-token prompt with `B = 8` were entirely cached, 2 blocks would match and no token would be left to run. The cap `(16 − 1) // 8 = 1` block leaves 8 tokens to compute. The cost is up to `B` recomputed tokens for a prompt that is a whole number of blocks.
2. Every prompt token is either computed in a prefill iteration or served from a cached block. The cache hits (1240) are tokens that did not need computing, so the computed tokens (1685) plus the hits equal the 2925 prompt tokens that the no-cache run computes. Test 8 asserts it for every run.
3. The cache holds references to blocks, so the pool has fewer blocks for running sequences, which makes running out of blocks (and so preemption) more likely: 7 against 4. But a preempted sequence that had finished its prefill finds its own prompt blocks in the cache when it is readmitted, so recomputation is smaller; prefill tokens fell from 3224 to 2338 (−886 tokens, −27%). More preemptions, cheaper preemptions.
4. It shows that the simulator makes the same admission, preemption, chunking and block-accounting decisions as an engine implementation that really allocates and frees blocks, in four configurations. It does not show that the cost constants are right (they are inputs to both), that the chosen workloads are realistic, or that a GPU engine behaves the same way.
5. The recomputed prefill would drop the tokens generated before the preemption, so the sequence would continue from the wrong context: outputs would differ from the reference in tests 6 and 9 (they have preemptions) and the timing in test 11's preemption configurations would differ too, because the prefill length changes. This is a prediction; to check it, change the line `v.target = len(v.prompt) + len(v.out)` in `preempt` and run the self-tests. Tests with no preemption (4, 5, 7, 8 roomy) would be unaffected.
6. Two sequences can share a partial block as long as neither writes to it. When one appends a token into the shared block, it must first copy the block into a new one and drop its reference to the original (Chapter 43's copy-on-write). The cost is one block copy per sequence that diverges, and the trie needs to key partial blocks by their token count as well as their contents.
7. For each prompt compare, at each generated position, the engine's top-1 token with the reference's. Report (a) the fraction of positions where they agree while following the *reference's* tokens (teacher forcing, so one divergence does not cascade), and (b) the maximum absolute difference in log-probability of the reference's chosen token. Require (a) ≥ 0.99 and (b) below a bound chosen from a run of the reference against itself with a different batch size. Treat a divergence where the reference's top two log-probabilities are within the (b) bound as a tie, not a failure.

## 46.12 Summary

- Part XIV built the core of an inference engine in four chapters: a paged KV cache, a continuous-batching scheduler with preemption, a prefix cache, and this engine joining them.
- Every answer matched the no-cache reference across seven configurations, including a pool small enough to force preemptions, chunked prefill, and a prefix cache all active together, with exact reference counts after every iteration. A deliberate corruption of the cache was caught.
- The Chapter 44 simulator's scheduling decisions matched the real implementation exactly.
- Caching and memory pressure interact: in a tiny pool the cache increased preemptions yet cut recomputation by 27%, so judge it by the work done, not by the preemption count.
- Equality of outputs is exact only because the computation is float64 and greedy. On a GPU the same checks need tolerances.
- None of it is a measurement of speed on a GPU. The runbook in 46.10 is how to find out.

## Files and sources

- Code: `docs/examples/inference-systems/ch46_mini_engine.py` (needs the Chapter 43, 44 and 45 files in the same folder).
- Background: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023); Yu et al., "Orca" (OSDI 2022); Zheng et al., "SGLang" (2023). This chapter implements its own simplified versions and makes no claim about the internals of vLLM or SGLang.
