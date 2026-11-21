# Final adversarial review: CuTe and CUTLASS Chapters 15–29

> Historical audit, scoped to its stated date and source revision. Later source corrections and outstanding GPU reruns are tracked in [the November 21 review repair record](../docs/review-fixes-2026-11-21.md). This note does not upgrade an old execution to a result for corrected source.

Review date: November 18, 2026. Reviewers were reassigned across the previous authoring groups to challenge accuracy, completeness, clarity and flow independently of their most recent contributions. The parent reviewed the resulting changes and integration.

## Findings fixed

- **P2 — Copy predicate preconditions (Chapter 20).** Explicitly stated matching logical domains, separate storage and pitch-specific allocation bounds. A source-derived predicate alone does not validate a smaller destination. Distinguished one-writer coverage from the dependencies of a partially overlapping copy; the lab does not implement memmove.
- **P2 — SIMT GEMM aliasing and extent scope (Chapter 23).** Stated separate A/B/C/D allocations, disjoint batch regions, fixed-size indexing bounds and the need for separate validation of an in-place C/D variant. Explained why a block barrier cannot protect inputs being overwritten across unrelated blocks.
- **P2 — CUTLASS support versus application memory validity (Chapter 24).** Clarified that can_implement does not establish pointer lifetime, allocation coverage, intended device access or valid aliasing. The caller must establish these conditions independently.

No additional substantive numerical, pinned-API, diagram or progression defect was identified in the reviewed material. This is a scoped review result, not a proof for arbitrary future kernels or inputs.

## Review coverage

- **15–19:** coordinate enumeration versus physical offsets; static/runtime and nested shapes; size/span/holes; coalesce, composition, complement and divided-mode grouping; tensor engines, transpose/slices, broadcasts and lifetime; partitioning, identity predicates, CTA/grid/batch formulas and byte conversion. Checked the pinned CuTe tutorials and implementation, both CPU arithmetic oracles and all six foundation worked SVGs.
- **20–24:** copy-tail ownership, pitches and guards; exact MMA traits and register-to-store mapping; bank models, padding and swizzles; bounded ragged GEMM reductions and epilogues; actual collective-builder argument ordering, schedule selection and adapter lifecycle. Recomputed diagram arithmetic and rendered all five kernel worked SVGs.
- **25–29:** pipeline start phases, participants, readiness/reuse and generation models against the pinned implementation; fusion operation order and runtime arguments; E2M1 packing, reconstruction and scale axes; timing scope/statistics; workbook bounds, tails, aliases and compression accounting. Executed the two inline precision snippets, recomputed worked results and reviewed all five advanced worked SVGs.

## Validation and limits

The tested executable sources and raw execution records were unchanged. No remote GPU host was accessed. Validation includes source/evidence identity checks, CPU arithmetic, SVG XML/font checks and visual inspection, strict MkDocs build, internal targets/fragments and displayed-source token preservation. Existing GPU execution evidence remains tied to its original exact sources, dates, hardware and commands.
