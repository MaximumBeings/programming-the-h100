# Final attention adversarial review — November 18, 2026

> Historical audit, scoped to its stated date and source revision. Later source corrections and outstanding GPU reruns are tracked in [the November 21 review repair record](../docs/review-fixes-2026-11-21.md). This note does not upgrade an old execution to a result for corrected source.

Baseline: `9f45d2d`. Three agents independently challenged Chapters 30–31, the CPU oracle/evidence checker, the Hopper collector, worked exercises and attention artwork. The parent reviewed integration and independently checked the backward row with finite differences and the raw/scaled softmax identities.

## Findings corrected

- **Softmax units:** Chapter 30 tracks a maximum of scaled scores. The pinned default FP16 collective instead keeps the unscaled QK maximum. Chapter 31 and its SVG now state the actual exp2 rescaling and `raw_max*scale_softmax + log(sum)` LSE expression. Finite additive bias must also enter in the correct units; zero/minus-infinity masks alone do not expose that distinction.
- **Collector false-positive scope:** matching only the number of success rows could accept a different problem or duplicated variants. Actual collection now checks the requested dimensions, direction/mask, one matching header, exact variant names/order, finite rates and positive finite verbose timings. A new regression test covers these cases.
- **CPU evidence semantics:** hashes alone did not ensure the command metadata described byte-compiling/running the recorded program or that stages occurred chronologically. The checker now verifies argv/source, interpreter/directory consistency, log association, source/system fields and ordered UTC intervals. Eight persistent tests include a valid fixture and seven rehashed invalid-metadata cases.
- **Clarity and completeness:** distinguished the CPU helper's Boolean allowed-mask/finite-input representation from mathematical negative-infinity notation. Added a numerical MHA/GQA/MQA head-sharing walkthrough, storage/address calculations and read-alias versus output/gradient ownership. The lab remains ordinary FP16 MHA; changing its H option is not a GQA/MQA switch. Prerequisite links now directly connect the preceding tensor, copy, MMA and pipeline lessons.

## Checks

All displayed attention scores, probabilities, O/LSE values, online states, positional masks, offsets and worked exercise answers were checked. The four attention figures and homepage overview were reviewed; the corrected ownership figure was rendered again. All 179 SVGs pass glyph coverage, and the 18 numerical worked figures pass font/license/bounds/overlap checks.

The CPU oracle passes normally and under optimized Python with identical output. Its current source and every archived CPU evidence file are unchanged. Nine collector tests and eight evidence tests pass. Collection planning was checked against the exact pinned checkout without executing CUDA. Strict MkDocs, internal links, displayed-source preservation and existing original/workshop evidence checks pass before publication.

No new CUDA compilation, GPU execution or profiler result is claimed. Hopper testing remains deferred at the user's request until capacity is available. The strengthened collector is ready to capture a separate record then.

## Pinned primary implementation

- [Runner, tensor strides and printed variant contracts](https://github.com/NVIDIA/cutlass/blob/1afc6d355b8cb992398f17beb4425702f1d57ddd/examples/88_hopper_fmha/88_hopper_fmha.cu).
- [Raw-maximum, rescaling and LSE implementation](https://github.com/NVIDIA/cutlass/blob/1afc6d355b8cb992398f17beb4425702f1d57ddd/examples/88_hopper_fmha/collective/fmha_collective_softmax.hpp).
- [Mask fusion contracts](https://github.com/NVIDIA/cutlass/blob/1afc6d355b8cb992398f17beb4425702f1d57ddd/examples/88_hopper_fmha/collective/fmha_fusion.hpp).
- [Advertised attention variants and head-sharing views](https://github.com/NVIDIA/cutlass/blob/1afc6d355b8cb992398f17beb4425702f1d57ddd/examples/88_hopper_fmha/README.md).
