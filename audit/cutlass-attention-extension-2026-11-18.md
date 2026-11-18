# CUTLASS attention extension — November 18, 2026

Chapters 30–31 connect the existing FlashAttention-3 design chapter and CuTe/CUTLASS course to the complete Hopper FMHA example 88 at CUTLASS v4.3.6 commit `1afc6d355b8cb992398f17beb4425702f1d57ddd`.

## Teaching and review

- Chapter 30 derives nonuniform Q/K/V scores, stable probabilities, complete outputs, online `(m,l,z)` updates, maximum-change rescaling, mask positions, layout/address units and exercises with worked answers.
- Chapter 31 maps those ideas to actual tensor strides, FMHA builders, stage ownership, mask specialization, O/LSE verification and FP16 backward. It includes explicit build, collection and profiling calls and worked derivative checks.
- Four new SVGs provide consistent chapter openers and worked illustrations; the homepage reading map, contents, setup and chapter handoffs include the extension.
- Authors independently reviewed each other's chapter. Numerical/API review used the pinned implementation, not an assumption of compatibility with current main. Visual review checked the new artwork. A collector timeout could leave compiler descendants; review caught it and the collector now terminates its own process group and preserves logs.
- A Chapter 12 troubleshooting table was narrowed: arbitrary wait-group settings, unconditional speedup/slowdown numbers and generic overlap prescriptions are not safe rules for every attention kernel.

## Validation and scope

The standard-library CPU oracle was byte-compiled and executed locally on November 18 UTC; exact commands, times, source hash, system information and raw logs are in `docs/validation/2026-11-18-attention-cpu`. `tools/verify_attention_walkthrough.py` enforces their identity and displayed output. These are CPU binary64 mathematical checks, not CUTLASS compilation or CUDA execution.

The new collector defaults to a plan and snapshots the exact pinned source before an explicitly requested build/run. It checks every expected verified variant and rejects zero-exit guards, skipped kernels, missing verification and failures. Eight host tests cover result classification, explicit case arguments and descendant cleanup on timeout. Plan metadata identifies the official runner source hash `b29f795d77867fbac7954c55c39042d7a5b145659f84742c568c305cfc47b4c5`.

No new example-88 GPU compilation, execution, sanitizer or profiler result was captured. The six forward cases and optional two backward cases remain ready for a separate Hopper record. Existing original/workshop program sources and execution records remain unchanged.

Validation includes CPU oracle/evidence checks, host collector tests, a dry collection plan against the pinned checkout, all existing source/evidence validators, SVG checks (179 assets; 18 numerical worked figures), strict MkDocs, internal links and displayed-source formatting. Publication is checked separately after push.

## Primary implementation

- [Pinned Hopper FMHA example and README](https://github.com/NVIDIA/cutlass/tree/1afc6d355b8cb992398f17beb4425702f1d57ddd/examples/88_hopper_fmha).
- [Pinned runner and reference comparisons](https://github.com/NVIDIA/cutlass/blob/1afc6d355b8cb992398f17beb4425702f1d57ddd/examples/88_hopper_fmha/88_hopper_fmha.cu).
- [Pinned causal and residual-mask implementations](https://github.com/NVIDIA/cutlass/blob/1afc6d355b8cb992398f17beb4425702f1d57ddd/examples/88_hopper_fmha/collective/fmha_fusion.hpp).
