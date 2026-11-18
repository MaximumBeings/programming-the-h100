# 15. CuTe — Layout Algebra You Can Check by Hand

[![CuTe maps logical coordinates to physical storage before assigning work to threads.](images/ch15/overview.svg)](images/ch15/overview.svg)

!!! tip "Your goal"

    Read a CuTe layout as an address function, distinguish a view from a transfer, and explain how a thread/value partition reaches an instruction operand. Review [shared-memory layout](04-cutensormap-tma-swizzling.md) and [WGMMA fragments](06-wgmma-part-1.md) first.

## 1 — The missing bridge to library kernels

A hand-written kernel often hides its layout in arithmetic: `m * leading_dimension + n`, a swizzle XOR, or a lane-dependent register index. CuTe makes such mappings explicit types and objects. That lets CUTLASS describe global tiles, shared-memory stages, copy ownership and MMA fragments using related operations instead of unrelated indexing formulas.

This chapter covers the **C++ CuTe API** distributed with CUTLASS **v4.3.6**, pinned in [Chapter 23](23-cutlass-gemm.md). CuTe’s Python DSL is a separate programming interface; do not mix its examples, imports or JIT rules with these C++ headers. Newer releases can add schedules and change builders, so this is a reproducible reading baseline rather than a claim that v4.3.6 is the latest release.

The primary references are the pinned [layout tutorial](https://github.com/NVIDIA/cutlass/blob/v4.3.6/media/docs/cpp/cute/01_layout.md), [layout algebra](https://github.com/NVIDIA/cutlass/blob/v4.3.6/media/docs/cpp/cute/02_layout_algebra.md), and [tensor tutorial](https://github.com/NVIDIA/cutlass/blob/v4.3.6/media/docs/cpp/cute/03_tensor.md).

## Before the algebra: a reproducible workspace

The introductory path is **15 → 16 → 17 → 18**. Continue through the intermediate copy/MMA/shared-memory/GEMM labs in **19–23**, then the advanced pipeline, fusion, precision and measurement chapters in **24–28**. You do not need to understand a CUTLASS builder before learning a coordinate map.

Use a terminal in the book repository root. On the CUDA host, prepare the same pinned dependency used throughout this sequence:

```bash
export CUTLASS_ROOT="$HOME/cutlass-v4.3.6"
bash tools/fetch_cutlass.sh "$CUTLASS_ROOT"
mkdir -p build/advanced
```

The helper verifies v4.3.6 commit `1afc6d355b8cb992398f17beb4425702f1d57ddd`. The CPU [indexing workbook](examples/advanced/indexing_workbook.py) needs only Python’s standard library, so you can work through the arithmetic on a laptop before using the GPU host. The CuTe C++ exercises use actual CUTLASS/CUDA headers; their compilation is separate from the Python reference’s checks.

## 2 — Shape says which coordinates; stride says where

For a two-mode layout with shape `(3,4)` and stride `(4,1)`, the address offset is

`offset(m,n) = 4m + n`, for `0 ≤ m < 3`, `0 ≤ n < 4`.

| Coordinate row | n=0 | n=1 | n=2 | n=3 |
| --- | ---: | ---: | ---: | ---: |
| m=0 | 0 | 1 | 2 | 3 |
| m=1 | 4 | 5 | 6 | 7 |
| m=2 | 8 | 9 | 10 | 11 |

The explicit construction is:

```cpp
auto row_major = cute::make_layout(
    cute::make_shape(cute::Int<3>{}, cute::Int<4>{}),
    cute::make_stride(cute::Int<4>{}, cute::Int<1>{}));
auto offset = row_major(cute::make_coord(2, 3)); // 11
```

Offsets are in the layout’s indexing units—normally elements when paired with a typed pointer—not automatically bytes. Multiply by `sizeof(Element)` when comparing with a byte-addressed PTX descriptor. `make_layout(shape)` defaults to a left-major compact layout; it is not a request for the row-major convention used above. Always print or enumerate a new layout before relying on a familiar name.

`Int<4>{}` carries the value in its type; `int k = 4` carries a runtime value. Both can participate in a layout. Static information lets template code prove constraints and specialize indexing, but it also increases the variety of instantiated types. Make dimensions static because an instruction or algorithm needs that fact, not simply because static syntax looks faster.

### Compare row-major and column-major without changing the coordinates

For the same shape `(3,4)`, column-major stride `(1,3)` gives this table:

| Coordinate row | n=0 | n=1 | n=2 | n=3 |
| --- | ---: | ---: | ---: | ---: |
| m=0 | 0 | 3 | 6 | 9 |
| m=1 | 1 | 4 | 7 | 10 |
| m=2 | 2 | 5 | 8 | 11 |

Both layouts have 12 logical elements and the same largest offset, but most interior coordinates name different storage positions. At `(1,2)`, row-major gives 6 and column-major gives 7. Testing only `(0,0)` and the last element would miss that difference. The words “row-major” refer to which mathematical coordinate changes contiguously; they do not follow from the order in which an author happened to list shape names.

For a row-major batched tensor `(L,M,N)` with packed batches, `offset(l,m,n)=l*M*N+m*N+n`. With physical row length `ld>=N` and batch stride `sb`, it becomes `l*sb+m*ld+n`. A tensor’s mathematical shape therefore does not uniquely determine its storage. A framework view can have padding or noncompact strides without changing its logical dimensions.

A transpose view swaps the coordinate interpretation and the corresponding shape/stride modes. It does not physically rearrange the allocation. On a rectangular 3×4 row-major matrix, a 4×3 transposed view uses stride `(1,4)`: transposed coordinate `(2,1)` refers to original `(1,2)`, offset 6. Keep the original allocation’s lifetime valid while any such view is used.

## 3 — Size and storage span are different

Now retain shape `(3,4)` but use stride `(8,1)`. The three rows start at offsets 0, 8 and 16. There are 12 logical coordinates, yet the largest referenced offset is 19. For this positive-stride example, the storage span through the last referenced element is 20.

- `size(layout)` describes the number of logical elements: 12.
- `cosize(layout)` describes the codomain span used for storage sizing in this example: 20.
- An external allocation may be larger still, for example to reserve a full padded row or several aligned stages.

A broadcast layout can map several coordinates to one offset. That may be useful for reads; it does not give independent destinations for parallel writes. The coordinate domain, image, storage bounds and ownership are separate properties. Do not generalize the simple positive-stride span arithmetic to arbitrary composed/swizzled layouts without checking the corresponding contract.

## 4 — Hierarchy is part of the meaning

A layout may have nested shapes such as `((2,2),4)`. The first mode retains a factorization: two subcoordinates describe its four positions. This is useful when those factors correspond to lanes, values per lane, subtiles or an instruction atom. Flattening the shape can hide exactly the structure that a copy or MMA partition expects.

`coalesce` simplifies compatible layout factors while preserving the mapping it represents. It does not move the tensor, erase data padding by copying, or preserve every original hierarchy boundary. When a later operation expects a particular mode structure, use the algebra’s mode/profile-aware forms and inspect the resulting shape.

## 5 — Composition is function composition

If layout `A` maps an index to a storage offset and layout `B` maps a new coordinate to an index accepted by `A`, their composition maps `x` to `A(B(x))`. Order matters. This is a view transformation: no loads or stores occur merely because a composed layout was constructed.

A simple scalar example makes the direction visible. Let `A(i)=2i` and `B(j)=j+1` over a valid domain. Then `A(B(j))=2j+2`, while `B(A(j))=2j+1`. Real CuTe layouts must satisfy the algebra’s compatibility rules; this arithmetic analogy does not make every arbitrary affine function a valid `Layout` constructor.

When debugging composition, enumerate the small domain on paper and compare three quantities: the input coordinate, its intermediate index, and its final offset. Then test the actual CuTe object. A type that compiles is not proof that the chosen mapping matches your matrix convention.

## 6 — Divide a layout into tiles

A contiguous 12-element layout can be logically divided into tiles of four. The new coordinates are `(within_tile, tile_number)`, with offsets `within_tile + 4 * tile_number`.

| Tile | Four offsets |
| --- | --- |
| 0 | 0, 1, 2, 3 |
| 1 | 4, 5, 6, 7 |
| 2 | 8, 9, 10, 11 |

```cpp
auto contiguous = cute::make_layout(cute::Int<12>{});
auto tiled = cute::logical_divide(contiguous, cute::Int<4>{});
auto offset = tiled(cute::make_coord(2, 1)); // 6
```

`logical_divide`, `zipped_divide` and their multidimensional relatives organize tile and remainder modes differently. Learn the returned shape, not just the function name. A layout view does not allocate a shared-memory stage, issue TMA, or predicate out-of-bounds coordinates. A kernel still needs those operations and their synchronization protocol.

## 7 — Tensors, slices and thread/value partitions

A CuTe tensor combines a data engine with a layout. A pointer-backed view over global memory differs from an owning register tensor even when their logical shapes match. Slicing chooses coordinates/modes; `local_tile` selects a tile; a tiled copy or tiled MMA assigns parts of a tensor to participating threads.

The useful debugging chain is:

**matrix coordinates → tile coordinates → thread/value coordinates → element offset → byte address**.

For a load, ask whether every needed element is covered. For a store, also ask whether two threads can target the same address. For MMA, ask whether the fragment matches the exact atom and operand convention. The register layouts from Chapters 6–7 are not replaced by a generic “row-major register array” merely because CuTe performs the partitioning.

When reading a `TiledCopy` or `TiledMma`, locate its atom, thread layout and value layout. Then inspect the per-thread slice and its partition functions. A partition describes ownership; the actual copy/MMA call performs work. NVIDIA’s [copy/MMA atom tutorial](https://github.com/NVIDIA/cutlass/blob/v4.3.6/media/docs/cpp/cute/0t_mma_atom.md) connects these layers.

## 8 — Swizzles are layout transforms, not faster copies

A shared-memory swizzle changes the mapping between logical coordinates and shared addresses. It can improve bank behavior for a particular access pattern. It must also agree with the TMA and WGMMA layout descriptions. A swizzle applied only to the producer or only to the consumer scrambles the computation.

Check the base alignment, pattern start, tile shape and descriptor fields from [Chapters 4 and 6](04-cutensormap-tma-swizzling.md). A swizzled allocation needs the correct storage span and alignment; `size` alone is not a universal shared-memory allocation recipe. Start with a supported CUTLASS builder layout before inventing a custom composition.

## 9 — Run the coordinate oracle

The complete [cute_layouts.cu](examples/advanced/cute_layouts.cu) enumerates both matrix mappings and the divided one-dimensional layout. It launches no GPU kernel. We compile it with the pinned CUDA/CUTLASS toolchain so that its C++ header dependencies match the later GEMM.

```bash
nvcc -std=c++17 -O2 --expt-relaxed-constexpr \
  -I "$CUTLASS_ROOT/include" \
  docs/examples/advanced/cute_layouts.cu -o build/advanced/cute_layouts
./build/advanced/cute_layouts
```

Create `build/advanced` and set `CUTLASS_ROOT` using the workspace steps above. Expected checks are the row-major coordinate table above, `size(padded)=12`, `cosize(padded)=20`, and `LAYOUT PASS` with zero errors. These are specifications to verify, not a captured run transcript. The separate [advanced collector](examples/advanced/README.md) retains actual output, source identities and system information for its executions.

??? question "Self-check: does a new layout transpose the allocation?"

    You change a pointer-backed tensor view’s strides. Have you transposed the bytes?

    **Answer:** No. You changed which bytes each logical coordinate names. A physical transpose requires reads and writes into the intended destination layout, with valid ownership and synchronization.

??? question "Self-check: how many elements do you allocate?"

    Shape `(3,4)`, stride `(8,1)`: is a 12-element buffer enough?

    **Answer:** No. Offset 19 is referenced, so this example needs a span of at least 20 elements. Additional padding/alignment may require a larger allocation.

## Continue in increasing depth

[Chapter 16](16-cute-layout-algebra.md) develops static/dynamic types, hierarchy, composition and division, and explains restrictions on inversion. [Chapter 17](17-cute-tensors-tiling.md) attaches layouts to tensor engines and builds safe tile views. [Chapter 18](18-cta-indexing.md) traces complete matrix coordinates through batch, CTA, K tile, thread and byte address. The [workbook](28-workbook-capstone.md) supplies worked answers rather than asking you to trust a printed PASS without knowing its oracle.

<!-- workshop:start:cute_layouts -->
## Complete source and captured execution

**Executed on November 18, 2026 (UTC).** Scope: **CPU execution of real CuTe coordinate/trait checks**.

Tested source SHA-256: `39e99111ba0dd6a48ecb812bd9ce6afe918e557407c8160a6dd8f8b1c530ab3f`. [Exact source](validation/2026-11-18-h100-workshop/cute_layouts.cu) · [System information](validation/2026-11-18-h100-workshop/system.json) · [Workshop results](cutlass-workshop-validation.md).

The listing below matches the tested source. Its presentation is formatted at build time; downloads retain the captured bytes.

```cuda
// CuTe coordinate checks. No GPU kernel is launched by this program.
#include <cute/tensor.hpp>
#include <cstdio>
#include <stdexcept>
using namespace cute;
int main() {
    auto row = make_layout(make_shape(Int<3>{}, Int<4>{}),
                           make_stride(Int<4>{}, Int<1>{}));
    auto padded = make_layout(make_shape(Int<3>{}, Int<4>{}),
                              make_stride(Int<8>{}, Int<1>{}));
    int errors = 0;
    for (int m = 0; m < 3; ++m) {
        for (int n = 0; n < 4; ++n) {
            const int offset = row(make_coord(m, n));
            errors += offset != 4 * m + n;
            errors += padded(make_coord(m, n)) != 8 * m + n;
            std::printf("%2d%s", offset, n == 3 ? "\n" : " ");
        }
    }
    errors += int(size(row)) != 12 || int(cosize(row)) != 12;
    errors += int(size(padded)) != 12 || int(cosize(padded)) != 20;
    auto contiguous = make_layout(Int<12>{});
    auto divided = logical_divide(contiguous, Int<4>{});
    for (int tile = 0; tile < 3; ++tile)
        for (int within = 0; within < 4; ++within)
            errors += divided(make_coord(within, tile)) != within + 4 * tile;
    std::printf("size(padded)=%d cosize(padded)=%d\n",
                int(size(padded)), int(cosize(padded)));
    std::printf("%s: 24 matrix coordinates, 12 tile coordinates; errors=%d\n",
                errors ? "LAYOUT FAIL" : "LAYOUT PASS", errors);
    return errors ? 1 : 0;
}
```

CUDA compilation UTC: `2026-11-18T14:09:13.582401+00:00`. Exit: `0`. [Compile log](validation/2026-11-18-h100-workshop/cute_layouts-compile.log).

Commands use scratch-directory filenames; raw logs retain exact paths. The collector’s CUTLASS include flags are configuration for that capture; the minimal standalone overlap/SIMT commands do not require those headers.

```bash
nvcc -std=c++17 -O2 -lineinfo -arch=sm_90a --expt-relaxed-constexpr -I "$CUTLASS_ROOT/include" -I "$CUTLASS_ROOT/tools/util/include" cute_layouts.cu -o ./example
```

Executed UTC: `2026-11-18T14:09:25.346704+00:00`. Exit: `0`. [Run log](validation/2026-11-18-h100-workshop/cute_layouts-run.log).

```bash
./example
```

**Actual captured output**

```text
 0  1  2  3
 4  5  6  7
 8  9 10 11
size(padded)=12 cosize(padded)=20
LAYOUT PASS: 24 matrix coordinates, 12 tile coordinates; errors=0
```

<!-- workshop:end:cute_layouts -->
