# CuTe/CUTLASS teaching coverage

Reviewed November 18, 2026. The scope is every CuTe/CUTLASS chapter, 15–29. Chapter 16 is the teaching baseline, not the only chapter with worked visuals. This matrix records a manual content review; SVG presence alone is not a test of teaching quality.

| Chapter | Worked illustration | Reasoning and code connection |
| --- | --- | --- |
| 15 Foundations | row-vs-column.svg | Static/runtime shapes and strides, selected coordinates, element/byte units and the coordinate oracle |
| 16 Strides | row-vs-column.svg; padding-and-storage.svg; transpose-view.svg | Logical grids, physical strips, padding, transpose, batches, broadcasts, bounds and exercises |
| 17 Layout algebra | composition-index-to-value.svg | Inner index, outer mapping, storage offset and stored value; coalesce, complement and division |
| 18 Tensor views | tile-identity-and-predicate.svg | Global-to-local coordinates, identity values, original bounds, retained strides and safe access |
| 19 CTA indexing | cta-operands-to-result.svg | Tile origins, local coordinates, each operand's address, complete reduction and result |
| 20 Tiled copy | kernel-copy-tail.svg | Six valid tail owners, source/destination pitches, values, byte offsets and predicates |
| 21 MMA fragments | kernel-mma-lane10.svg | Four register entries, trait encoding, logical outputs and independent padded stores |
| 22 Shared layouts | kernel-shared-banks.svg | Word addresses, banks, padding/XOR and the difference between permutation and transaction claims |
| 23 Tiled GEMM | kernel-gemm-ragged.svg | Ragged owner, partial K dots, running accumulator, barriers and one-time epilogue |
| 24 Collective builders | kernel-builder-contract.svg | Real alias assembly, physical B versus CuTe modes, runtime arguments and checked adapter lifecycle |
| 25 Pipelines | ch25-stage-generations.svg | Slots, generations, separate phase conventions, readiness, completed readers, release and drain |
| 26 Fusion | ch26-fusion-values.svg | Independent alpha/beta branches, numeric join, activation, storage and counterexamples |
| 27 Precision | ch27-nibbles-scales-dot.svg | Values, E2M1 codes, low/high nibbles, block scales, reconstruction and dot products |
| 28 Measurement | ch28-event-join-boundaries.svg | Event dependencies, joined completion, setup/steady-state boundaries and worked statistics |
| 29 Workbook | ch29-boundary-tile.svg | Boundary tile coverage, invalid-coordinate aliasing, K tails and worked exercise progression |

The illustrations are adjacent to their matching derivations. Explanations identify notation and units, connect the actual API or explicitly labeled model to intermediate calculations, and explain checks and tempting wrong answers. Derived examples are distinguished from captured execution; existing source-specific commands, dates, output and systems remain intact.

Validation includes arithmetic assertions in the generators, rendered visual inspection, SVG XML/font checks, strict MkDocs build, internal links, displayed-code token preservation and immutable execution-evidence checks. No tested CUDA source was changed and no H100 instance was used for this expansion.
