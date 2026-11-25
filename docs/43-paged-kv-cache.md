# Chapter 43 — A Paged KV Cache from Scratch

[![Chapter 43: Logical tokens, physical blocks](images/chapter-openers/43-inference.svg)](images/chapter-openers/43-inference.svg)


!!! warning "Evidence status"
    **Run on CPU (NumPy):** the block allocator, the paged cache with fork and copy-on-write, paged attention with an online softmax, 17 original self-tests including a randomised property test, plus boundary regressions, and four experiments (`ch43_paged_kv_cache.py`).
    **Not run on any GPU.** Nothing here measures speed. The memory results are exact counts; the length distribution (lognormal, median 300 tokens, clipped to 2048) is **assumed**, not taken from a real workload. The vLLM commands in 43.9 are **UNTESTED** and written from memory of the CLI: check every flag against `vllm serve --help` for your version.

This is the first chapter of Part XIV, where we build the core of an inference engine in the style of vLLM and SGLang, one piece at a time, with every piece tested on a CPU. The chapter numbers start at 43 because Chapters 41 and 42 are still reserved for Part XIII.

Chapter 32 gave the anatomy of an engine. Here we build its memory manager.

## Worked system map

[![Logical tokens, physical blocks: step-by-step hand trace](images/worked/ch43-inference.svg)](images/worked/ch43-inference.svg)


The November 25 correction validates K/V shape and conversion **before** allocation or copy-on-write, and rejects negative/out-of-range block indices. See the [fresh execution record](validation-inference-2026-11-25.md).

## 43.1 The problem the KV cache creates

During generation each sequence keeps a key and a value vector per token, per layer, per KV head. For a model with `L` layers, `h_kv` KV heads, head dimension `d`, and `b` bytes per element:

`bytes per token = 2 × L × h_kv × d × b`

Hand arithmetic, BF16 (`b = 2`), using the published Llama-3 shapes (8 KV heads, head dimension 128; 32 layers for the 8B model, 80 for the 70B model):

| Model | Bytes per token | One 2048-token reservation |
|---|---|---|
| 8B | 2 × 32 × 8 × 128 × 2 = 131 072 B = 128 KiB | 256 MiB |
| 70B | 2 × 80 × 8 × 128 × 2 = 327 680 B = 320 KiB | 640 MiB |

(Check the layer and head counts against your model's `config.json`.) The script prints the same numbers.

The cache grows by one token per step, and nobody knows in advance how long a sequence will be. The naive design reserves one contiguous region of `max_len` tokens per sequence. That wastes everything between the real length and `max_len`, and it also fragments memory as sequences of different sizes come and go.

Paging, the idea behind vLLM's PagedAttention, borrows from virtual memory: cut the cache into fixed-size **blocks** of `B` tokens, give each sequence a **block table** that lists its blocks in order, and let the blocks live anywhere in the pool.

```
sequence s (len 10, B = 4)         physical pool (blocks)
  block table: [7, 2, 5]            0 [ ........ ]
   logical block 0 -> physical 7    1 [ ........ ]
   logical block 1 -> physical 2    2 [ tok 4-7  ]  <- full
   logical block 2 -> physical 5    3 [ ........ ]
                                    5 [ tok 8-9 . .]  <- 2 of 4 slots used
                                    7 [ tok 0-3  ]  <- full
```

Only the last block can be partly empty, so the waste per sequence is at most `B − 1` slots.

## 43.2 The allocator

`BlockAllocator` keeps a free list and a **reference count** per block. `alloc` pops a block and sets its count to 1; `incref` adds a sharer; `decref` removes one and returns the block to the free list at zero. Two defensive rules do real work:

- `decref` on a block with count 0 raises (a double free).
- `incref` on a free block raises (a use-after-free).

These two checks catch the most common memory-manager bugs at the moment they happen rather than as silent corruption later.

## 43.3 The cache: block tables, append, fork

`PagedKVCache` owns two pools, `K` and `V`, each shaped `(n_blocks, B, heads, dim)`, plus a table `sid → {table, len}`.

**Append** writes one token at slot `len % B` of the last block:

1. If `len % B == 0`, the last block is full (or the sequence is empty): allocate a new block.
2. Otherwise, if the last block is shared (refcount above 1), **copy it first** (copy-on-write): allocate a new block, copy the old contents, drop one reference to the old block, and point the table at the copy.
3. Write the token.

**Fork** gives a new sequence the parent's table and increments every block's refcount. It copies no data, so forking costs `O(blocks)` of bookkeeping and zero bytes of KV.

**Atomicity.** Both allocation points (new block, copy-on-write copy) happen *before* any state is changed, so an `OutOfBlocks` leaves the cache exactly as it was (tests 11 and 12). A scheduler can therefore catch the exception and preempt something without having to repair a half-finished append.

## 43.4 Paged attention

The attention for one decode query must read the sequence's whole cache. Gathering it into one contiguous array would copy the very bytes paging avoids moving. `paged_attention` instead walks the block table and keeps a running softmax (the same online trick as flash attention): for each block it updates a running maximum `m`, running denominator `l` and running weighted sum `acc`, rescaling the old values when the maximum grows.

```
for each block b in table:
    scores = q · K[b]ᵀ / √d
    m' = max(m, max(scores))
    rescale = exp(m − m')
    l   = l·rescale + Σ exp(scores − m')
    acc = acc·rescale + exp(scores − m') · V[b]
    m = m'
result = acc / l
```

Test 13 compares this against ordinary dense attention in float64 for `B ∈ {1, 2, 5, 16}` and lengths from 1 to 100: the largest difference is 4.4e-16. Test 14 checks that the answer does not depend on where the blocks sit in the pool, and test 15 that a fork attends identically to its parent.

A GPU kernel does the same walk with the block table in global memory; making it fast (coalesced loads, one block per thread block, split-K for long sequences) is a kernel problem outside this chapter.

## 43.5 Self-tests (17, all pass)

| # | What is checked |
|---|---|
| 1–3 | allocator: every block handed out once and recovered; double free and use-after-free raise; empty pool fails cleanly |
| 4 | append then gather returns exactly the appended data, for `B = 1, 3, 4, 16` |
| 5 | freeing a sequence returns all of its blocks |
| 6 | fork allocates nothing; tables are identical |
| 7 | child append copies the shared partial block; the parent's data is unchanged and exactly one block was added |
| 8 | the same, with the parent appending after the fork |
| 9 | fork exactly at a block boundary shares everything and the next append adds one fresh block, not a copy |
| 10 | once the parent is freed, the child's refcount is 1 and it appends in place |
| 11–12 | an append, and a copy-on-write append, that hit `OutOfBlocks` change nothing |
| 13–15 | paged attention equals dense attention, regardless of block placement and across forks |
| 16 | **property test:** 20 random sessions of 400 operations (add, append, fork, free, with a deliberately small pool so OOM occurs) against a plain-list reference model; after every operation the invariants hold and every sequence's K and V equal the reference |
| 17 | the contiguous first-fit allocator used in Experiment 4 reuses freed holes |

`check_invariants` verifies after each operation that every block's refcount equals the number of table entries pointing at it, that a block is on the free list exactly when its refcount is zero, that the free list has no duplicates, and that each table has `ceil(len / B)` entries. A leak or a double-free shows up as a failure of that check.

## 43.6 Experiment 1 — internal waste versus block size

200 000 sequence lengths (lognormal, median 300, clipped to [1, 2048], mean 433.1):

| B | measured waste per sequence (slots) | theory (B − 1)/2 | waste as % of live tokens | block-table entries per sequence |
|---|---|---|---|---|
| 1 | 0.00 | 0.00 | 0.00% | 433.1 |
| 4 | 1.47 | 1.50 | 0.34% | 108.7 |
| 8 | 3.44 | 3.50 | 0.79% | 54.6 |
| 16 | 7.37 | 7.50 | 1.70% | 27.5 |
| 32 | 15.23 | 15.50 | 3.52% | 14.0 |
| 64 | 30.92 | 31.50 | 7.14% | 7.3 |
| 128 | 62.00 | 63.50 | 14.31% | 3.9 |
| 256 | 126.33 | 127.50 | 29.17% | 2.2 |

The waste per sequence tracks `(B − 1)/2`, the average of a uniformly distributed last-block fill, to within about 3%. The gap comes from clipping at lengths 1 and 2048, which are not uniform modulo `B`. **The cost of a big block is internal waste; the cost of a small block is a long table to walk, and on a GPU, less contiguous reading per block.** The table-length column is only bookkeeping size: how it affects kernel speed is not measured here.

## 43.7 Experiment 2 — how many sequences fit

A pool of 65 536 token slots.

| scheme | sequences that fit | vs contiguous |
|---|---|---|
| contiguous, reserve 2048 tokens each | 32 | 1.0× |
| paged, B = 1 | 150.0 | 4.7× |
| paged, B = 16 | 147.5 | 4.6× |
| paged, B = 64 | 139.8 | 4.4× |
| paged, B = 256 | 116.3 | 3.6× |

(mean over 200 trials of filling the pool with fresh lengths). The 4.6× gain at `B = 16` is almost entirely **not reserving `max_len`**: with a mean length of 433 and a reservation of 2048, a contiguous scheme holds about 21% live data. The result depends on the assumed length distribution: if real sequences were close to `max_len` the gain would shrink toward 1×. Measure your own length distribution (Chapter 36's harness records lengths).

## 43.8 Experiment 3 — sharing a prompt with fork

A prompt of `P` tokens is prefilled once, forked into `n` children, the parent is freed, and each child generates `g` tokens. Blocks used, by simulation and by formula:

`blocks = ⌊P/B⌋ + n × ⌈((P mod B) + g) / B⌉`

(the full blocks of the prompt are shared; the partial prompt block is copied by every child but the last, which inherits it in place, so each child owns `⌈((P mod B) + g)/B⌉` blocks counting that one).

| B | P | n | g | blocks (fork) | formula | blocks (no sharing) | saving |
|---|---|---|---|---|---|---|---|
| 16 | 512 | 8 | 64 | 64 | 64 | 288 | 77.8% |
| 16 | 520 | 8 | 64 | 72 | 72 | 296 | 75.7% |
| 16 | 1000 | 16 | 128 | 206 | 206 | 1136 | 81.9% |
| 64 | 1000 | 16 | 128 | 63 | 63 | 288 | 78.1% |
| 16 | 100 | 4 | 400 | 110 | 110 | 128 | 14.1% |

The simulation matches the formula in every row (the script asserts it). Sharing helps when the shared prompt is long compared with what each child adds: 78–82% for long prompts, 14% when the children generate four times as many tokens as the prompt held. The no-sharing column is `n × ⌈(P + g)/B⌉`.

**Hand check, row 1:** `⌊512/16⌋ = 32` shared blocks, plus `8 × ⌈(0 + 64)/16⌉ = 32`: 64 blocks. Without sharing: `8 × ⌈576/16⌉ = 288`. Saving `1 − 64/288 = 77.8%` ✓.

This is the mechanism behind parallel sampling and beam search. Chapter 45 generalises it to prefixes shared across *different* requests.

## 43.9 Experiment 4 — external fragmentation

A contiguous allocator can refuse a request even though enough total space is free, because the free space is split into holes. Paging cannot, since any free block is usable. The stream: one arrival per step, a 50% chance that one random sequence departs, 20 000 steps, a pool of 65 536 slots, lengths with median 200. Both systems saturate.

| | refused | refused although total free space was enough | mean live tokens / pool (after warm-up) |
|---|---|---|---|
| contiguous (first-fit) | 9492 | **9492** | 0.893 |
| paged, B = 16 | 9557 | **0** | 0.950 |

**This is the best case for the contiguous scheme:** it allocates exactly the final length, which a real server does not know in advance (Experiment 2 is the realistic comparison). Even so, every one of its refusals happened with enough total free space, which is fragmentation by definition, and it holds 5.7 points less live data than the paged cache. The paged cache's refusals are genuine: it was full to within less than one request.

## 43.10 Failure modes and edge cases

| Situation | What happens | Handled by |
|---|---|---|
| Pool exhausted mid-append | `OutOfBlocks`, state unchanged | atomic append (tests 11–12); the scheduler must preempt |
| Free of an unknown or already-freed sequence | `KeyError` / double-free `ValueError` | allocator checks |
| Fork then both children append | each copies the last shared block once; the last owner writes in place | copy-on-write (tests 7, 8, 10) |
| Fork at a block boundary | no copy at all | test 9 |
| Prefix longer than the pool | prefill fails partway | scheduler must check `blocks_needed` before admitting (Chapter 44) |
| Block size 1 | zero internal waste, but tables as long as the sequences | Experiment 1 |
| Sequence length 0 | empty table, `gather` returns empty arrays | `gather` guard |
| Writing into a shared block without copying | silently corrupts the other sequence | the refcount check in `append`; test 16 would catch it |

**Not modelled:** multiple layers and the per-layer pools (each layer has its own K and V, but they share the block table, so the table and the allocator are per sequence, not per layer); GPU memory-bandwidth effects; swapping blocks to CPU (Chapter 34 covers offloading); sliding-window or other attention variants that free old blocks; quantised KV cache.

## 43.11 On real hardware (UNTESTED)

These commands are from memory; verify every flag against your installed version.

1. Print the engine's own accounting: start the server and read the startup log for the number of GPU blocks and the KV cache size in tokens.

   `vllm serve MODEL --block-size 16 --gpu-memory-utilization 0.90 --max-model-len 4096`

2. Compare the logged capacity with your own arithmetic from 43.1: `free memory after weights × utilization ÷ (bytes per token × B)` blocks. A difference larger than a few percent means activations or other buffers are taking memory you did not count.
3. Repeat with `--block-size 8`, `16` and `32` (check which values your version accepts) and record the capacity and, with the Chapter 36 client, the throughput at a fixed rate. Experiment 1 predicts the capacity change; the throughput change is a kernel effect this chapter cannot predict.
4. Record the real distribution of `prompt + output` lengths from your traffic, and redo Experiment 2 with it.

## 43.12 Exercises

1. A 70B BF16 model on 8 H100s leaves 40 GiB per GPU for KV cache after weights and activations, with the 8 KV heads sharded one per GPU (tensor parallel 8). How many tokens fit in total, and how many 2048-token contiguous reservations?
2. Why is the waste per sequence about `(B − 1)/2` and not `B − 1`? When would it be larger?
3. Show that a fork at a block boundary followed by one append in each child uses two new blocks, while a fork in the middle of a block uses two as well, but for a different reason.
4. Extend `PagedKVCache` with `truncate(sid, n)` for speculative decoding rollback. What must happen to blocks that become empty, and to a block that is shared?
5. A scheduler admits a request only if the pool has `⌈prompt/B⌉` free blocks. Give a case where it then fails mid-generation, and say what it should do.
6. Experiment 4 gives the contiguous allocator the true final length. Predict, without running it, what happens to its refusals and utilisation if it must reserve `max_len = 2048` instead.

### Solutions

1. Per token across the group: 327 680 B in total, 40 960 B per GPU (one KV head per GPU: 2 × 80 × 1 × 128 × 2 = 40 960). 40 GiB = 42 949 672 960 B ÷ 40 960 B = 1 048 576 tokens (that is, 2²⁰). Contiguous 2048-token reservations: 1 048 576 ÷ 2048 = 512 sequences. Paged, with the Experiment 1 length distribution (mean 433.1) and `B = 16` (average waste 7.4 slots): about `1 048 576 / (433.1 + 7.4)` ≈ 2380 sequences, which is 4.6× as in Experiment 2.
2. The last block's fill is roughly uniform over 1..B, so the unused part is uniform over 0..B−1, with mean `(B − 1)/2`. It is larger if lengths are concentrated: if every sequence is exactly one token past a block boundary the waste per sequence is `B − 1`. Experiment 1's small shortfall (7.37 vs 7.50) comes from clipping.
3. Keep the parent alive. At a boundary, each child's first append finds no partial block and allocates a fresh one: two new blocks. Mid-block, the last block has three owners (parent and two children), so each child copies it: again two new blocks, and the parent keeps the original. The counts agree, the reasons differ. If the parent were freed first, the mid-block case would need only one new block, because the second child would be the last owner and write in place.
4. `truncate` sets `len = n`, then drops the references to table entries beyond `⌈n/B⌉` (a `decref` frees those whose count reaches zero). The new last block, if shared, is left shared: the next append will copy it (copy-on-write), so no extra step is needed. Stale data in the freed tail of the last block is harmless because `len` hides it, but it must never be read by attention, which uses `len` already.
5. Admission checked prompt blocks only; each running sequence also grows by one block every `B` steps. Ten sequences each admitted with their exact prompt blocks may exhaust the pool when several cross a block boundary together. The scheduler should reserve headroom for growth or be ready to preempt: free a victim's blocks and recompute or swap it later (Chapter 44).
6. Reserving 2048 each lets the pool hold 32 sequences, so once 32 are live nearly every arrival is refused. Utilisation would be about 32 × 300 / 65 536 ≈ 0.15 (mean length is about 300 for median 200, sigma 0.9). Fragmentation becomes rare because all regions have the same size, but the pool is mostly reserved-and-empty instead. This is a prediction, not a measured result: run it to check.

## 43.13 Summary

- A paged KV cache removes two costs of contiguous reservation: waste from reserving `max_len` (4.6× more sequences fit in this model workload) and external fragmentation (zero genuine refusals when paged).
- The internal waste is bounded by one block per sequence and averages `(B − 1)/2`; choosing `B` is a trade between that waste and table length/kernel efficiency.
- Reference counts plus copy-on-write make fork free and safe; the last owner writes in place.
- Making allocation atomic lets a scheduler treat "out of memory" as a recoverable event, which the next chapter builds on.
- Everything here is CPU and counts. Whether a given block size is fast depends on the kernel and hardware, and has to be measured.

## Files and sources

- Code: `docs/examples/inference-systems/ch43_paged_kv_cache.py`
- The idea comes from the vLLM paper, "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., SOSP 2023). This chapter implements its own version of the idea and makes no claim about vLLM's internals.
