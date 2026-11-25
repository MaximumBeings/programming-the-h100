# Inference chapters: agent audit and complete code — November 25, 2026

Chapters 32–47 and the FP4 appendix now display **18 complete Python files directly on their pages**, with run commands, supporting-module links, a reading order and a [source bundle](examples/inference-systems/inference-systems.zip). Each listing is checked against the downloadable file and the rendered HTML, including indentation and the complete entry point. The shorter explanatory excerpts remain in place before the full implementation.

Three agents reviewed the core measurement/distribution chapters, the gateway chapters, and the engine/FP4 chapters. They examined source code, worked arithmetic, failure cases, explanations, diagrams and the scope of the execution evidence. A separate cross-review of the corrected gateway found and fixed another causal ordering error before publication.

## Findings and corrections

| Chapters | Confirmed problem | Correction and check |
|---|---|---|
| 32 | Occupancy integration omitted requests arriving during an iteration; a zero batch cap could prevent progress. | Integrate their waiting time and reject invalid configurations. A two-request regression checks the exact occupancy identity. |
| 33 | A failed Gloo child could leave the parent waiting forever. | Shared deadline, exit-code checks, rank ownership checks and guaranteed child cleanup. Tests deliberately crash and stall workers and check that none remain. |
| 34 | Active KV offload was declared impossible; the overlap arithmetic mixed TP2 step time with unsharded KV bytes. | State conditional feasibility and per-rank/link assumptions. Evenly sharded TP2 KV uses 2,048 bytes/token/layer/rank; the illustrative overlap threshold becomes about 7,050 tokens. |
| 35 | Layerwise transfer could incorrectly have zero exposed latency, ignoring the final layer's availability. | Model layer availability and the serialized link. At 4,096 tokens and the assumed 400 Gb/s link, the last-layer tail is about 0.3355 ms. Preserve request identity and output-work counts. |
| 36 | Failure drain time was omitted from throughput/cost; timeouts lacked terminal timestamps; SSE events were treated as token counts. | Count terminal failure time in the measurement window, always close/timestamp failed requests, and require reported completion-token usage for a fixed-token success. Socket regressions cover all three cases. The request-IID bootstrap table is explicitly identified as an uncalibrated diagnostic for correlated serving requests. |
| 37 | Draft KV memory was omitted from “feasible” batches. | Account for target and draft KV on the most loaded GPU. At 2k context the model's maximum batches become BF16+1B: 7; FP8+1B: 94; FP8+8B: 40. Recompute the tables and replace infeasible worked cases. |
| 37 | Sampling and communication conclusions were overstated or reversed. | Exact rejection sampling must use the actual proposal and target distributions; equal sampling hyperparameters are not required. Shared target overhead can raise the relative speedup ratio while worsening absolute latency. |
| 38 | Invalid vectors could mutate cache state; a zero-error cache was called an infinite economic win even when embedding cost exceeded savings. | Validate before mutation and handle the no-benefit break-even case explicitly. Clarify cache memory and clustering assumptions. |
| 39 | Exactly masked logits could produce NaN entropy; the signed quality-gap formula and confidence claims were wrong. | Use the limiting value of zero-probability entropy terms, reject undefined all-masked rows, fix the sign and distinguish pointwise intervals from post-selection guarantees. |
| 40 | Unstarted requests at the simulation horizon could count as SLO successes. | Keep censored requests in the denominator without counting them as successful starts. |
| 41 | A request could change the breaker with its future outcome before an overlapping arrival was processed. | Deliver events chronologically; refresh scenario tables and add an overlapping-request regression. |
| 42 | Cached answers could exist before their producing request completed; routing and breaker results also leaked future state. | Use a chronological event loop, completion-based cache insertion, real prefix completion and delayed breaker notifications. Replace request-IID confidence intervals with paired run summaries. |
| 42 cross-review | Direct-small first content could precede the prefix that decides whether the small answer is used. | Hold first content until the routing decision is ready. A deterministic test checks the timing boundary. |
| 43–46 | Invalid sizes, budgets, token IDs, arrivals and position lengths could reach allocation/scheduling; empty attention returned NaN. | Reject invalid input before state mutation and reject an empty visible-key set. Regression tests check ownership remains intact after failure. |
| 43–46 | Refusal, pin-lifetime, fork-cost and reference-independence claims exceeded what the examples establish. | Distinguish capacity refusals from external fragmentation; a `CacheFull` matched length is not a pinned KV handle. Qualify timing and shared-reference-helper limitations. |
| 47 / FP4 | Indexing, trusted-input and historical-capture boundaries needed clearer wording. | Make indices explicitly zero-based, state that the CUDA probe consumes trusted inputs, and explain FP4 emulation's rounding policy. The real-vLLM result remains tied to its archived client. |

The review also corrected portable run commands, section references, statistical caveats and worked answers. The offload and disaggregation SVGs now state compatible bandwidth/transfer assumptions, including the final-layer tail. No historical raw output was edited to make a correction appear tested.

## Fresh execution record

The follow-up captures use **Python 3.12.14, NumPy 2.5.5 and PyTorch 2.8.0 on macOS 26.6 arm64**. These are CPU runs, including local Gloo processes and loopback mock HTTP servers; they are not new H100 measurements.

The final collection contains **34 successful commands**:

- 30 invocations covering the focused checks and complete demonstrations of all 16 CPU reference programs, including the complete benchmark mock sweep and four-rank Gloo example;
- four regression suites: the earlier boundary tests plus the core, gateway and engine adversarial tests;
- exact source snapshots, command lines, start/end times, exit codes and unmodified stdout/stderr.

Inspect the [result index](validation/2026-11-25-inference-review/results.json), [system details](validation/2026-11-25-inference-review/system.json), [dependency lock](validation/2026-11-25-inference-review/requirements.lock.txt) and [checksum inventory](validation/2026-11-25-inference-review/checksums.json). Representative raw captures:

| Check | Raw log |
|---|---|
| Core adversarial regressions: 11 tests | [Core review](validation/2026-11-25-inference-review/logs/test_inference_core_review-regression.log) |
| Gateway numerical/causality regressions: six groups | [Gateway review](validation/2026-11-25-inference-review/logs/test_inference_gateway_review-regression.log) |
| Engine/cache boundaries and independent numerical comparison: 10 tests | [Engine review](validation/2026-11-25-inference-review/logs/test_inference_engine_review-regression.log) |
| Complete corrected Gloo demonstration | [Chapter 33](validation/2026-11-25-inference-review/logs/ch33_moe_expert_parallel-full.log) |
| Complete corrected benchmark demonstration | [Chapter 36](validation/2026-11-25-inference-review/logs/ch36_benchmark_teardown-full.log) |
| Corrected speculation-capacity tables | [Chapter 37](validation/2026-11-25-inference-review/logs/ch37_spec_decode_70b-full.log) |
| Corrected chronological gateway experiments | [Chapter 42](validation/2026-11-25-inference-review/logs/ch42_gateway_capstone-full.log) |
| Mini-engine reference comparisons | [Chapter 46](validation/2026-11-25-inference-review/logs/ch46_mini_engine-full.log) |

The worker-cleanup regressions intentionally crash or stall children; an expected child error in that raw log is part of the negative test. The regression command itself must complete successfully.

## A host warning that was investigated rather than hidden

The older NumPy 2.2.6 environment emitted matrix-multiply overflow/divide warnings on this Apple M4 host. Reviewers checked that the tiny transformer's outputs were finite and compared its complete forward pass with an independent non-BLAS `einsum` implementation at seven sequence lengths. Maximum absolute difference was **6.22e-15**. The newer reference environment repeated the comparison without those warnings.

This is consistent with the [NumPy M4 warning report](https://github.com/numpy/numpy/issues/28687) and [follow-up report](https://github.com/numpy/numpy/issues/29820). Warnings were not globally suppressed, and the finite-output/independent-path regression remains in the test suite. This is an environment observation, not a claim that every overflow warning is harmless.

## Evidence boundaries and remaining hardware work

The [earlier November 25 H100 record](validation-inference-2026-11-25.md) is preserved unchanged at the raw-file level. Its CUDA comparisons and real-vLLM streams still establish what ran with those archived sources. The follow-up CPU fixes are supported by their separate current-source captures.

**No CUDA kernel changed in this review.** The updated shared Chapter 36 HTTP client has fresh CPU protocol, accounting and complete mock-server validation. Its exact-current integration with real vLLM has **not** been repeated after those client changes. Do not treat the archived serving capture as that later integration run.

Real multi-GPU MoE, RDMA disaggregation, 70B speculative decoding, native B200 FP4, production offload, a live autoscaling cluster and real cache/router quality studies still need their own hardware/workload validation. They are not represented by these CPU simulations.

## Reproduce and keep the pages in sync

From the repository root, use the [reference environment guide](examples/inference-systems/README.md). To collect new CPU evidence, choose a new output directory; the collector refuses to overwrite an existing run:

```bash
python tools/collect_inference_review.py --output /tmp/inference-review-rerun
python tools/embed_inference_sources.py --check
python tools/verify_inference_review.py
python tools/verify_inference_validation.py
mkdocs build --strict
python tools/verify_inline_inference.py --site-dir site
```

The collector runs programs; the verifiers check the committed records and current source. A later source change requires a new capture and an explicit evidence transition. The original inventory, new inventory, complete listings, bundle and rendered HTML are all checked in the publication workflow.
