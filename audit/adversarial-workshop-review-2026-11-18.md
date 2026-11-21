# Adversarial review of the expanded CuTe/CUTLASS course

> Historical audit, scoped to its stated date and source revision. Later source corrections and outstanding GPU reruns are tracked in [the November 21 review repair record](../docs/review-fixes-2026-11-21.md). This note does not upgrade an old execution to a result for corrected source.

Reviewed November 18, 2026. Three agent reviewers covered technical correctness, comprehensiveness, robustness, beginner explanations and course progression. The parent reviewed integration, diagrams, chapter migration and evidence consistency.

## Findings resolved

- Beginner entry unnecessarily implied prior WGMMA/TMA mastery. CPU coordinate work now has a separate entry path; hardware prerequisites apply when the relevant kernels are introduced.
- Shapes could be mistaken for runtime access guards. Added an explicit invalid-coordinate counterexample and the need for independent bounds checks.
- GPU pointer tags could imply host accessibility. Distinguished view construction, valid address-space access, constness, producer completion and allocation lifetime.
- Different B storage conventions could be carried between chapters. Explicitly connected column-major indexing examples to the later row-major SIMT baseline.
- Fragment ownership alone omitted collective participation. Added convergence and tail-handling requirements for warp MMA.
- Exact equality lacked its bounded arithmetic justification. Derived the binary-rational and reduction-size limits for the tested SIMT inputs.
- Passing a baseline could imply arbitrary strides, aliases, zero sizes or other modes were covered. Added explicit extension contracts and independent validation requirements.
- Activation tests needed to distinguish numerical coverage from omitted stores. Explained positive/negative/zero preactivation categories and separate sentinel checks.
- Precision prose overstated BF16 validation. The record now names the executed FP16 labs; BF16 is a separately validated extension.
- Quantization examples were initially lossless and used only one K block. Added lossy nearest-even rounding and a worked two-block scale calculation.
- Repeated in-place beta*C benchmarks could change the problem between samples. Added state-reset and repeated-operation accounting requirements.
- Timing examples omitted a concrete distribution decision. Added ordered samples, median/range/mean and explicit scope distinctions.
- Workbook K=35 packing could look directly supported by the complete-block encoder. Distinguished the paper tail contract and the padding required for an implementation.
- Traffic accounting needed an explicit output datatype. Separated FP16-D and FP32-D byte counts.
- New terminology was absent from the homepage glossary. Added linked CuTe/CUTLASS, stride, ownership, fusion and precision entries.
- Striding was dispersed rather than taught as a dedicated prerequisite. Added Chapter 16 and a separately executed CPU address oracle; renumbered the remaining course through Chapter 29.
- Overview art was insufficient for the requested teaching approach. Added three worked SVGs with logical values, physical offset strips and selected coordinates, plus inline numerical panels throughout the new chapters.

## Integration and evidence

Existing published chapter routes redirect directly to their new locations, preserving section hashes. Source generators, navigation, homepage curriculum, cross-links and chapter labels agree with the new numbering.

The ten frozen workshop sources and original CUDA source suite were not changed. No remote GPU host was used for this review. The new stride program was byte-compiled and executed locally, with system information, UTC dates, output and source hash recorded separately under docs/validation/2026-11-18-strides.

The evidence verifier now checks the separate layout/swizzle probes, strict profiler provider and successful CSV row, trace-analysis identities and stride records, in addition to the original immutable dataset.

Validation: strict MkDocs build; internal files/fragments and redirect targets; source-format preservation; SVG XML and font readability; original execution registries; expanded workshop inventory; CPU stride/workbook/precision models. Hardware behavior beyond the recorded tests is not established by this offline review.
