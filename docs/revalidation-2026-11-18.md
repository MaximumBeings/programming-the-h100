# November 18 rerun — two H100 GPUs

**27 programs compiled; 26 workloads passed.** The eight-GPU training program compiled but its workload requires eight visible GPUs, so it was not executed on this two-GPU host. Its separately captured [eight-A100 numerical result](eight-gpu-validation.md) remains valid.

The four corrected barrier-ordering programs passed, with zero-error memcheck records. The attention recurrence passed all 12 cases, and the stronger Chapter 7 nonuniform-coordinate variant passed again. This is an additional execution date; earlier raw records remain unchanged.

[System information](validation/2026-11-18-two-h100/system-info.log) · [Execution manifest](validation/2026-11-18-two-h100/results.json) · [Integrity inventory](validation/2026-11-18-two-h100/integrity.json).

**System:** 2× H100 80GB HBM3, CC 9.0, NV18 peer topology; Ubuntu 24.04.5 LTS; driver 580.105.08; CUDA compiler 12.8.93; GCC 13.3.0; C++17, `-O2`, `sm_90a`. All timestamps below are UTC.

## Program results

### ex01_cluster_inspect.cu

**PASS** · Compile UTC: `2026-11-18T11:19:10.753247+00:00` · Executed UTC: `2026-11-18T11:19:13.520596+00:00`.

Source SHA-256: `ff9c3f47e5bc98d502ca02011672ee2fe024f52e93614d3eb91284953ce6e2f3`.

[Exact source](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex01_cluster_inspect/ex01_cluster_inspect.cu) · [Compile log](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex01_cluster_inspect/compile.log) · [Run log](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex01_cluster_inspect/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_cluster_inspect.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════╗
║  Thread Block Cluster Inspection — H100 Demo         ║
╚══════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)

Per-Block Report:
Block(global) Cluster Rank   Cluster Size   My SMEM Val  Partner Val  Exchange OK?
--------------------------------------------------------------------------------
  0            0              2              0            100          ✓ PASS
  1            1              2              100          0            ✓ PASS
  2            0              2              200          300          ✓ PASS
  3            1              2              300          200          ✓ PASS

DSMEM Exchange Tests: 4/4 passed

Cluster Metadata Verification:
  ✓ All cluster metadata correct

════════════════════════════════════════════════
ALL TESTS PASSED — Clusters working correctly!
════════════════════════════════════════════════
```

### ex02_ptx_basics.cu

**PASS** · Compile UTC: `2026-11-18T11:19:14.175173+00:00` · Executed UTC: `2026-11-18T11:19:15.109469+00:00`.

Source SHA-256: `50b26d6098753bb1cf202e4482308bf86020d3e0eee4fad0b59f2cfcc2eef70f`.

[Exact source](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex02_ptx_basics/ex02_ptx_basics.cu) · [Compile log](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex02_ptx_basics/compile.log) · [Run log](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex02_ptx_basics/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex02_ptx_basics.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════╗
║  Inline PTX Basics — Full Test Suite                  ║
╚══════════════════════════════════════════════════════╝

Test 1: add.u32 (42 + 10 = 52)
  ✓ PASS: result = 52

Test 2: mad.lo.u32 (5*7+3 = 38)
  ✓ PASS: result = 38

Test 3: fma.rn.f32 (2.5 * 4.0 + 1.5 = 11.5)
  ✓ PASS: result = 11.5000

Test 4: predicate max
  ✓ PASS: max(10,20) = 20
  ✓ PASS: max(30,5) = 30

Test 5: special registers (%tid.x, %ctaid.x, %laneid)
  ✓ PASS: tid.x = 0
  INFO: ctaid.x = 1 (block index, not checked — scheduling dependent)
  INFO: laneid = 0 (should be 0 since tid=0)

Test 6: bitwise ops (a=0xFF00, b=0x0F0F)
  ✓ PASS: and = 0x00F00
  ✓ PASS: or = 0x0FF0F
  ✓ PASS: xor = 0x0F00F
  ✓ PASS: shl = 0xFF000
  ✓ PASS: shr = 0x00FF0

Test 7: early clobber — (7+3)*(7-3) = 40
  ✓ PASS: result = 40

Test 8: read-write operand — 100 + 3 = 103
  ✓ PASS: result = 103

Test 9: immediate operand template
  ✓ PASS: 1 << 5 = 32
  ✓ PASS: 7 << 3 = 56

Test 10: volatile + memory clobber (store 0xDEADBEEF to SMEM)
  ✓ PASS: loaded 0xDEADBEEF from SMEM

Test 11: vote.ballot (even lanes → 0x55555555)
  ✓ PASS: ballot = 0x55555555

╔══════════════════════════════════════════════════════╗
║  Results: 17/17 tests passed                          ║
║  ✓ ALL PASS                                          ║
╚══════════════════════════════════════════════════════╝
```

### ex03_state_spaces.cu

**PASS** · Compile UTC: `2026-11-18T11:19:15.678429+00:00` · Executed UTC: `2026-11-18T11:19:16.526377+00:00`.

Source SHA-256: `1ae7bdef0dbe25d98cc52b89f78a3b0562def59c985a9a8db1286c79bee47fc8`.

[Exact source](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex03_state_spaces/ex03_state_spaces.cu) · [Compile log](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex03_state_spaces/compile.log) · [Run log](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex03_state_spaces/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex03_state_spaces.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════╗
║  PTX State Spaces — Test Suite                        ║
╚══════════════════════════════════════════════════════╝

Test 2: .shared — Thread 0 reads thread 1's slot (expected 2.5)
  ✓ PASS: thread 0 read 2.5 from thread 1's slot

Test 3: .global — Multiply by 10 via explicit PTX
  ✓ PASS: all 8 values multiplied by 10 correctly

Test 4: .const — Read SCALE_FACTOR[0] = 2.0 via PTX and C++
  ✓ PASS: PTX read = 2.0, C++ read = 2.0 (both = 2.0)

Test 6: Combined — global→SMEM→compute with SCALE_FACTOR[0]=2.0
  ✓ PASS: 128 elements scaled by 2.0 (sample: in=5 out=10)

╔══════════════════════════════════════════════════════╗
║  Results: 4/4 tests passed                            ║
╚══════════════════════════════════════════════════════╝
```

### ex04_ptx_pointers.cu

**PASS** · Compile UTC: `2026-11-18T11:19:17.076551+00:00` · Executed UTC: `2026-11-18T11:19:18.162592+00:00`.

Source SHA-256: `77953a91b667b873d64296e0437bb2ca69234210f9dbdb9998125ed3a7512c8c`.

[Exact source](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex04_ptx_pointers/ex04_ptx_pointers.cu) · [Compile log](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex04_ptx_pointers/compile.log) · [Run log](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex04_ptx_pointers/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex04_ptx_pointers.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════╗
║  ex04: PTX Pointers, State Spaces & Packed Types      ║
╚══════════════════════════════════════════════════════╝

Test 1: cvta + ld.shared (smem[0]=42, smem[1]=99)
  ✓ PASS: loaded 42.0 and 99.0, SMEM offset=0x400

Test 2: st.shared/ld.shared neighbor (expect 7.0 and 17.0)
  ✓ PASS: my=7.0 neighbor=17.0

Test 3: ld.global/st.global multiply by 2
  ✓ PASS: all elements×2 correct

Test 4: atom.shared.add (64 threads → counter=64)
  ✓ PASS: counter=64

Test 5: Packed FP16 add (1+3=4, 2+4=6)
  ✓ PASS: slot0=4.0 slot1=6.0

Test 6: mov.b32 bit reinterpret (float↔uint32)
  ✓ PASS: pi bits=0x40490FD0, roundtrip matches

╔══════════════════════════════════════════════════════╗
║  6/6 tests passed   ✓ ALL PASS                    ║
╚══════════════════════════════════════════════════════╝
```

### ex05_ptx_datatypes.cu

**PASS** · Compile UTC: `2026-11-18T11:19:18.726938+00:00` · Executed UTC: `2026-11-18T11:19:20.021963+00:00`.

Source SHA-256: `711a86340850bdbae0dfd8b9d809ad27e7c6da2b057253d8f667a698e9630220`.

[Exact source](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex05_ptx_datatypes/ex05_ptx_datatypes.cu) · [Compile log](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex05_ptx_datatypes/compile.log) · [Run log](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex05_ptx_datatypes/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex05_ptx_datatypes.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════╗
║  ex05: PTX Data Types & Packed Operations             ║
╚══════════════════════════════════════════════════════╝

Test 1: cvt FP32→FP16 (1.5f → 0x3E00)
  ✓ PASS: 0x3E00

Test 2: FP16 pack/unpack (1.0f=0x3C00, 2.0f=0x4000)
  ✓ PASS: packed=0x40003C00 lo=0x3C00 hi=0x4000

Test 3: add.bf16x2 (2.0+3.0=5.0 in both slots)
  ✓ PASS: lo=5.00 hi=5.00

Test 4: add.f64 (1.23456789012345 + 9.87654321098765)
  ✓ PASS: 11.111111101111101

Test 5: mad hi+lo (0xFFFFFFFF * 0xFFFFFFFF)
  ✓ PASS: lo=0x00000001 hi=0xFFFFFFFE

╔══════════════════════════════════════════════════════╗
║  5/5 tests passed   ✓ ALL PASS                    ║
╚══════════════════════════════════════════════════════╝
```

### ex06_comprehensive.cu

**PASS** · Compile UTC: `2026-11-18T11:19:20.571501+00:00` · Executed UTC: `2026-11-18T11:19:23.333448+00:00`.

Source SHA-256: `4c5fb40a537b3c5dfc2190023302f953f98867349e486a5daeba6e73e823be8d`.

[Exact source](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex06_comprehensive/ex06_comprehensive.cu) · [Compile log](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex06_comprehensive/compile.log) · [Run log](validation/2026-11-18-two-h100/02-clusters-inline-ptx/ex06_comprehensive/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex06_comprehensive.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════════╗
║  ex06: Comprehensive Capstone — Clusters + PTX + DSMEM   ║
╚══════════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)
       ✓ Hopper/Blackwell GPU detected — clusters supported.

Launch config:
  Grid:    2 blocks
  Block:   64 threads
  Cluster: 2 blocks per cluster (1 cluster total)
  N:       128 elements

Verification (first 8 and last 4 elements):
  Index    Input        Expected     Got          OK?
  ------------------------------------------------------------
  0        0.0          64.0         64.0         ✓
  1        1.0          66.0         66.0         ✓
  2        2.0          68.0         68.0         ✓
  3        3.0          70.0         70.0         ✓
  4        4.0          72.0         72.0         ✓
  5        5.0          74.0         74.0         ✓
  6        6.0          76.0         76.0         ✓
  7        7.0          78.0         78.0         ✓
  63       63.0         190.0        190.0        ✓
  64       64.0         64.0         64.0         ✓
  65       65.0         66.0         66.0         ✓
  126      126.0        188.0        188.0        ✓
  127      127.0        190.0        190.0        ✓

  ✓ ALL 128 elements correct!

  What just happened (review):
  ├── Block 0 loaded input[0..63] into its SMEM (PTX ld.global)
  ├── Block 1 loaded input[64..127] into its SMEM (PTX ld.global)
  ├── cluster.sync() ensured both blocks' SMEM was ready
  ├── Each block read the partner's SMEM via map_shared_rank (DSMEM)
  ├── PTX fma.rn.f32 computed the pairwise sum
  └── Results written back to global memory (PTX st.global)

╔══════════════════════════════════════════════════════════╗
║  ✓ ALL TESTS PASSED — Comprehensive example complete!     ║
╚══════════════════════════════════════════════════════════╝
```

### ex01_mbarrier_basic.cu

**PASS** · Compile UTC: `2026-11-18T11:19:23.992648+00:00` · Executed UTC: `2026-11-18T11:19:24.852132+00:00`.

Source SHA-256: `406ff5009b3c3ef7db54ee6660fc4c1159bd75790b7ae24f1196ae8d651664c7`.

[Exact source](validation/2026-11-18-two-h100/03-asynchronicity-mbarrier/ex01_mbarrier_basic/ex01_mbarrier_basic.cu) · [Compile log](validation/2026-11-18-two-h100/03-asynchronicity-mbarrier/ex01_mbarrier_basic/compile.log) · [Run log](validation/2026-11-18-two-h100/03-asynchronicity-mbarrier/ex01_mbarrier_basic/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_mbarrier_basic.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════════╗
║  ex01: mbarrier Pipeline — Init/Arrive/Wait/Reuse        ║
╚══════════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)
Config: K=4 tiles, 256 elements/tile, 64 threads/block

Element checks: 1024 / 1024 passed

╔══════════════════════════════════════════════════════════╗
║  ✓ ALL PASS — mbarrier pipeline working correctly!     ║
╚══════════════════════════════════════════════════════════╝
```

Compute Sanitizer: **ERROR SUMMARY: 0 errors**. [Memcheck log](validation/2026-11-18-two-h100/correction-memcheck/03-asynchronicity-mbarrier/ex01_mbarrier_basic/memcheck.log) · [Command and timestamps](validation/2026-11-18-two-h100/correction-memcheck/03-asynchronicity-mbarrier/ex01_mbarrier_basic/memcheck.json).

### ex02_war_prevention.cu

**PASS** · Compile UTC: `2026-11-18T11:19:25.401355+00:00` · Executed UTC: `2026-11-18T11:19:26.237306+00:00`.

Source SHA-256: `d0d3b587edde9a7c77812263f13b5c29de73d5d8e26e9d9dc068e828b0761395`.

[Exact source](validation/2026-11-18-two-h100/03-asynchronicity-mbarrier/ex02_war_prevention/ex02_war_prevention.cu) · [Compile log](validation/2026-11-18-two-h100/03-asynchronicity-mbarrier/ex02_war_prevention/compile.log) · [Run log](validation/2026-11-18-two-h100/03-asynchronicity-mbarrier/ex02_war_prevention/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex02_war_prevention.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════════╗
║  ex02: WAR Prevention — Two-Barrier Pipeline             ║
╚══════════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)
Config: 8 tiles × 128 elements = 1024 total
Element errors: 0 / 1024
Sample: in=7 → out=14 (expected 14)

╔══════════════════════════════════════════════════════════╗
║  ✓ ALL PASS — WAR hazard correctly prevented!          ║
╚══════════════════════════════════════════════════════════╝
```

Compute Sanitizer: **ERROR SUMMARY: 0 errors**. [Memcheck log](validation/2026-11-18-two-h100/correction-memcheck/03-asynchronicity-mbarrier/ex02_war_prevention/memcheck.log) · [Command and timestamps](validation/2026-11-18-two-h100/correction-memcheck/03-asynchronicity-mbarrier/ex02_war_prevention/memcheck.json).

### ex01_tma_basic.cu

**PASS** · Compile UTC: `2026-11-18T11:19:26.803033+00:00` · Executed UTC: `2026-11-18T11:19:27.695435+00:00`.

Source SHA-256: `c94e84ae2a4cf9dbd8d2c0c8a65c40399613eca6a5a40b262109c9a2f5f57caa`.

[Exact source](validation/2026-11-18-two-h100/04-cutensormap-tma-swizzling/ex01_tma_basic/ex01_tma_basic.cu) · [Compile log](validation/2026-11-18-two-h100/04-cutensormap-tma-swizzling/ex01_tma_basic/compile.log) · [Run log](validation/2026-11-18-two-h100/04-cutensormap-tma-swizzling/ex01_tma_basic/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_tma_basic.cu -o ./example -lcuda
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════════╗
║  ex01: cuTensorMap Setup + TMA Load + Verify             ║
╚══════════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)
Descriptor setup:
  Matrix:       256 × 128 FP32
  Tile:         32 × 32 = 1024 elements = 4096 bytes
  globalDim:    {256, 128}
  globalStride: {1024} bytes
  boxDim:       {32, 32}
  swizzle:      128B
  l2Promotion:  128B

cuTensorMapEncodeTiled: SUCCESS

Launching 32 tile blocks (8×4 tiles)...
Element checks: 32768 / 32768 passed

╔══════════════════════════════════════════════════════════╗
║  ✓ ALL PASS — TMA loaded all tiles correctly!          ║
╚══════════════════════════════════════════════════════════╝
```

### ex02_tma_3d.cu

**PASS** · Compile UTC: `2026-11-18T11:19:28.240111+00:00` · Executed UTC: `2026-11-18T11:19:29.363398+00:00`.

Source SHA-256: `ad620dba57d0780b5d18871d8593e783a2b269b6442958110c398daaf8ab0173`.

[Exact source](validation/2026-11-18-two-h100/04-cutensormap-tma-swizzling/ex02_tma_3d/ex02_tma_3d.cu) · [Compile log](validation/2026-11-18-two-h100/04-cutensormap-tma-swizzling/ex02_tma_3d/compile.log) · [Run log](validation/2026-11-18-two-h100/04-cutensormap-tma-swizzling/ex02_tma_3d/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex02_tma_3d.cu -o ./example -lcuda
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════════╗
║  ex02: 3D TMA — Attention Tensor [B][S][D] Tile Load     ║
╚══════════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)
3D Tensor: [4][128][64] FP16
Tile:      [1][32][64] = 2048 elements = 4096 bytes

Descriptor created successfully.

Launching 16 blocks (4 seq_tiles × 4 batches)...
Element checks: 32768 / 32768 passed

╔══════════════════════════════════════════════════════════╗
║  ✓ ALL PASS                                              ║
╚══════════════════════════════════════════════════════════╝
```

Compute Sanitizer: **ERROR SUMMARY: 0 errors**. [Memcheck log](validation/2026-11-18-two-h100/correction-memcheck/04-cutensormap-tma-swizzling/ex02_tma_3d/memcheck.log) · [Command and timestamps](validation/2026-11-18-two-h100/correction-memcheck/04-cutensormap-tma-swizzling/ex02_tma_3d/memcheck.json).

### ex01_bulk_raw.cu

**PASS** · Compile UTC: `2026-11-18T11:19:29.952145+00:00` · Executed UTC: `2026-11-18T11:19:30.798016+00:00`.

Source SHA-256: `f355b8990f51c2e2f2784e0d693d90653acf4e07e36eae714b1ee434aa92a805`.

[Exact source](validation/2026-11-18-two-h100/05-cp-async-bulk-l2-hinting/ex01_bulk_raw/ex01_bulk_raw.cu) · [Compile log](validation/2026-11-18-two-h100/05-cp-async-bulk-l2-hinting/ex01_bulk_raw/compile.log) · [Run log](validation/2026-11-18-two-h100/05-cp-async-bulk-l2-hinting/ex01_bulk_raw/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_bulk_raw.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════════╗
║  ex01: Raw cp.async.bulk + mbarrier + L2 evict_last      ║
╚══════════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)
Config: 6 tiles × 1024 FP32 = 4096 bytes per tile

Checks: 6144 / 6144 passed

╔══════════════════════════════════════════════════════════╗
║  ✓ ALL PASS                                              ║
╚══════════════════════════════════════════════════════════╝
```

Compute Sanitizer: **ERROR SUMMARY: 0 errors**. [Memcheck log](validation/2026-11-18-two-h100/correction-memcheck/05-cp-async-bulk-l2-hinting/ex01_bulk_raw/memcheck.log) · [Command and timestamps](validation/2026-11-18-two-h100/correction-memcheck/05-cp-async-bulk-l2-hinting/ex01_bulk_raw/memcheck.json).

### ex02_reduce_async.cu

**PASS** · Compile UTC: `2026-11-18T11:19:31.351016+00:00` · Executed UTC: `2026-11-18T11:19:32.175941+00:00`.

Source SHA-256: `52617e1a22a4619a1fd65dc92d4eac529b3bd02ee2ea34803ab51213f829f39a`.

[Exact source](validation/2026-11-18-two-h100/05-cp-async-bulk-l2-hinting/ex02_reduce_async/ex02_reduce_async.cu) · [Compile log](validation/2026-11-18-two-h100/05-cp-async-bulk-l2-hinting/ex02_reduce_async/compile.log) · [Run log](validation/2026-11-18-two-h100/05-cp-async-bulk-l2-hinting/ex02_reduce_async/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex02_reduce_async.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════════╗
║  ex02: cp.reduce.async — Atomic Tile Reduction           ║
╚══════════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)
Config: 4 blocks each adding their value (1,2,3,4) to 256 elements
Expected result: every element = 1+2+3+4 = 10.0

Errors: 0 / 256 elements
Sample: acc[0]=10.0 acc[127]=10.0 acc[255]=10.0

╔══════════════════════════════════════════════════════════╗
║  ✓ ALL PASS — cp.reduce.async works correctly!         ║
╚══════════════════════════════════════════════════════════╝
```

### ex01_wgmma_basic.cu

**PASS** · Compile UTC: `2026-11-18T11:19:32.743235+00:00` · Executed UTC: `2026-11-18T11:19:34.146676+00:00`.

Source SHA-256: `282064c18eedf80395d5c5ed41bc0d1e4160c8d3886bf831c90b68628b84ad04`.

[Exact source](validation/2026-11-18-two-h100/06-wgmma-part-1/ex01_wgmma_basic/ex01_wgmma_basic.cu) · [Compile log](validation/2026-11-18-two-h100/06-wgmma-part-1/ex01_wgmma_basic/compile.log) · [Run log](validation/2026-11-18-two-h100/06-wgmma-part-1/ex01_wgmma_basic/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_wgmma_basic.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════════╗
║  ex01: wgmma.mma_async — m64n64k16 BF16→FP32            ║
╚══════════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)
Shape: m64n64k16, D_REGS=32 per thread, 128 threads total
Patterned BF16 inputs; every accumulator checked against CPU GEMM.

Threads with finite accumulators: 128 / 128
D register values correct: 4096 / 4096
Max absolute error: 0.00000000
Sample d[0] = -0.246094 (expected -0.246094)

╔══════════════════════════════════════════════════════════╗
║  ✓ ALL PASS — wgmma computes correct BF16→FP32 result!  ║
╚══════════════════════════════════════════════════════════╝
```

### ex01_wgmma_pipeline.cu

**PASS** · Compile UTC: `2026-11-18T11:19:34.697322+00:00` · Executed UTC: `2026-11-18T11:19:36.086020+00:00`.

Source SHA-256: `76960b5ead233cd05431fe9c4c7847f859e3270a7407461f4dbe91006c1c2790`.

[Exact source](validation/2026-11-18-two-h100/07-wgmma-part-2/ex01_wgmma_pipeline/ex01_wgmma_pipeline.cu) · [Compile log](validation/2026-11-18-two-h100/07-wgmma-part-2/ex01_wgmma_pipeline/compile.log) · [Run log](validation/2026-11-18-two-h100/07-wgmma-part-2/ex01_wgmma_pipeline/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_wgmma_pipeline.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════════╗
║  ex01: wgmma commit/wait pipeline + stmatrix epilogue    ║
╚══════════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)
Shape: m64n64k16, FP16 accumulator, 2 tiles pipelined
Expected: all acc values = K×0.01² = 0.001600

Threads with non-zero sums in BOTH tiles: 128 / 128
Sample out_A[0] = 0.001600
Output verification: 0 errors across both 64×64 tiles

╔══════════════════════════════════════════════════════════╗
║  ✓ ALL PASS — commit/wait pipeline + stmatrix working!  ║
╚══════════════════════════════════════════════════════════╝
```

### ex01_roofline.cu

**PASS** · Compile UTC: `2026-11-18T11:19:36.642155+00:00` · Executed UTC: `2026-11-18T11:19:37.502670+00:00`.

Source SHA-256: `d4e66adac175c64e13c8761c308a1a5c2151164a5505cd6959d04b8abab61449`.

[Exact source](validation/2026-11-18-two-h100/08-kernel-design-part-1/ex01_roofline/ex01_roofline.cu) · [Compile log](validation/2026-11-18-two-h100/08-kernel-design-part-1/ex01_roofline/compile.log) · [Run log](validation/2026-11-18-two-h100/08-kernel-design-part-1/ex01_roofline/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_roofline.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════╗
║  Roofline Model Demo — Memory vs Compute Bound        ║
╚══════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)
     Peak memory bandwidth (theoretical): 3352.3 GB/s
     SM count: 132

════════════════════════════════════════════════════════
KERNEL A: elementwise_scale (expected: MEMORY-BOUND)
════════════════════════════════════════════════════════
  Elements: 64M    Bytes moved: 537 MB    AI: 0.125 FLOP/Byte
  (Ridge point for FP32 CUDA cores ≈ 20 FLOP/Byte)
  → AI=0.125 << 20 → MEMORY-BOUND ✓

  Time: 0.226 ms
  Bandwidth achieved: 2375.9 GB/s
  GFLOPS achieved:    297.0
  → Kernel is BANDWIDTH limited (not compute limited)

════════════════════════════════════════════════════════
KERNEL B: tile_matmul (expected: COMPUTE-BOUND)
════════════════════════════════════════════════════════
  M=N=K=512    FLOPs: 268.4 M    Bytes: 3.1 MB    AI: 85.3 FLOP/Byte
  → AI=85.3 >> 20 → COMPUTE-BOUND ✓

  Time: 0.033 ms
  GFLOPS achieved: 8019.7
  Effective BW:    94.0 GB/s (not the bottleneck)
  → Kernel is COMPUTE limited (bandwidth not saturated)

╔══════════════════════════════════════════════════════╗
║  KEY TAKEAWAY:                                        ║
║  • elementwise_scale: optimize bandwidth, not FLOPs   ║
║  • tile_matmul: optimize FLOPs, pipeline depth        ║
║  • Wrong optimization → wrong results → wasted time   ║
╚══════════════════════════════════════════════════════╝
```

### ex02_warp_spec.cu

**PASS** · Compile UTC: `2026-11-18T11:19:38.058112+00:00` · Executed UTC: `2026-11-18T11:19:38.924950+00:00`.

Source SHA-256: `fb1010dd4b8597fc35c1f99944db4a44e052228546c7f1e794f78560043be675`.

[Exact source](validation/2026-11-18-two-h100/08-kernel-design-part-1/ex02_warp_spec/ex02_warp_spec.cu) · [Compile log](validation/2026-11-18-two-h100/08-kernel-design-part-1/ex02_warp_spec/compile.log) · [Run log](validation/2026-11-18-two-h100/08-kernel-design-part-1/ex02_warp_spec/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex02_warp_spec.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════╗
║  Warp Specialization Demo                             ║
║  256 threads = WG0 (producer) + WG1 (consumer)       ║
╚══════════════════════════════════════════════════════╝

Launch: 1 block × 256 threads
  WG0 (threads   0-127): PRODUCER role
  WG1 (threads 128-255): CONSUMER role

Role Assignment Check:
  WG0 role recorded: 0 (expected 0 = producer)
  WG1 role recorded: 1 (expected 1 = consumer)
  Role assignment: ✓ CORRECT

Consumer Output Verification:
  Expected formula: output[i] = 2 × (3×i + 7) = 6i + 14
  Checking all 128 outputs...

  Index    Expected     Got          OK?
  ────────────────────────────────────────
  0        14.0         14.0         ✓
  1        20.0         20.0         ✓
  2        26.0         26.0         ✓
  3        32.0         32.0         ✓
  4        38.0         38.0         ✓
  ... (checking all 128, showing first 5 and last 3) ...
  125      764.0        764.0        ✓
  126      770.0        770.0        ✓
  127      776.0        776.0        ✓

════════════════════════════════════════════════════════
Results: 128/128 output values correct
Role assignment: CORRECT

╔══════════════════════════════════════════════════════╗
║  ✓ ALL TESTS PASSED                                  ║
║                                                      ║
║  KEY OBSERVATIONS:                                   ║
║  • WG0 (threads 0-127) executed ONLY producer code   ║
║  • WG1 (threads 128-255) executed ONLY consumer code  ║
║  • No warp divergence within either warpgroup        ║
║  • __syncthreads() ensured safe handoff              ║
║  • In production: replace sync with mbarrier (§9)    ║
╚══════════════════════════════════════════════════════╝
```

### ex03_setmaxnreg.cu

**PASS** · Compile UTC: `2026-11-18T11:19:39.476370+00:00` · Executed UTC: `2026-11-18T11:19:40.281281+00:00`.

Source SHA-256: `c640beecccfe53b9968d7e5f14e939dfaa0966c45829c480694bfa45556e2292`.

[Exact source](validation/2026-11-18-two-h100/08-kernel-design-part-1/ex03_setmaxnreg/ex03_setmaxnreg.cu) · [Compile log](validation/2026-11-18-two-h100/08-kernel-design-part-1/ex03_setmaxnreg/compile.log) · [Run log](validation/2026-11-18-two-h100/08-kernel-design-part-1/ex03_setmaxnreg/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex03_setmaxnreg.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════╗
║  setmaxnreg Demo — Dynamic Register Resizing         ║
╚══════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)

Register Plan:
  WG0 (threads   0-127): setmaxnreg.dec → 32 regs (producer)
  WG1 (threads 128-255): setmaxnreg.inc → 128 regs (consumer)

WG0 (PRODUCER, 32 regs after resize):
  ✓ All 128 values correct (registers NOT corrupted by setmaxnreg)
  Sample: output[0]=1.0 (expect 1.0), output[1]=3.5 (expect 3.5)

WG1 (CONSUMER, 128 regs after resize):
  ✓ All 128 values correct (registers NOT corrupted by setmaxnreg)
  Sample: output[0]=2.0 (expect 2.0), output[1]=6.5 (expect 6.5)

╔══════════════════════════════════════════════════════╗
║  ✓ ALL TESTS PASSED                                  ║
║  setmaxnreg correctly resized registers without      ║
║  corrupting existing register state.                 ║
╚══════════════════════════════════════════════════════╝
```

### ex04_circular_buffer.cu

**PASS** · Compile UTC: `2026-11-18T11:19:40.843154+00:00` · Executed UTC: `2026-11-18T11:19:41.816510+00:00`.

Source SHA-256: `3ea5fd796a9fa77c9d2c002bdf8a440c9071f248946514164134ea66c41c4665`.

[Exact source](validation/2026-11-18-two-h100/08-kernel-design-part-1/ex04_circular_buffer/ex04_circular_buffer.cu) · [Compile log](validation/2026-11-18-two-h100/08-kernel-design-part-1/ex04_circular_buffer/compile.log) · [Run log](validation/2026-11-18-two-h100/08-kernel-design-part-1/ex04_circular_buffer/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex04_circular_buffer.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════╗
║  Circular Pipeline Buffer Simulation                  ║
║  N_STAGES=4, DEPTH=2, K_TILES=8                    ║
╚══════════════════════════════════════════════════════╝

[Prologue]: filling first 2 stages
  k=0: Producer writes stage 0  |  Consumer: waiting
  k=1: Producer writes stage 1  |  Consumer: waiting

[Steady State]: producer and consumer overlap
  k=2: Producer writes stage 2  |  Consumer reads stage 0
  k=3: Producer writes stage 3  |  Consumer reads stage 1
  k=4: Producer writes stage 0  |  Consumer reads stage 2
  k=5: Producer writes stage 1  |  Consumer reads stage 3
  k=6: Producer writes stage 2  |  Consumer reads stage 0
  k=7: Producer writes stage 3  |  Consumer reads stage 1

[Drain]: flushing 2 remaining stages
  drain 0: Consumer reads stage 2  |  Producer: done
  drain 1: Consumer reads stage 3  |  Producer: done

════════════════════════════════════════════════════════
Stage Assignment Verification:
  Expected producer: 0,1,2,3,0,1,2,3,-1,-1 (then drain)
  Expected consumer: -1,-1,0,1,2,3,0,1,2,3

  Iter    Prod    Cons
  ────────────────────
     0       0      -1 ✓
     1       1      -1 ✓
     2       2       0 ✓
     3       3       1 ✓
     4       0       2 ✓
     5       1       3 ✓
     6       2       0 ✓
     7       3       1 ✓
     8      -1       2 ✓
     9      -1       3 ✓

Output Verification (element 0 and element 10):
  output[0]  = 280.000 (expected 280.000) ✓
  output[10] = 280.800 (expected 280.800) ✓

╔══════════════════════════════════════════════════════╗
║  Stage assignment errors: 0                           ║
║  Runtime barrier errors:  0                           ║
║  ✓ ALL TESTS PASSED — circular buffer correct        ║
╚══════════════════════════════════════════════════════╝
```

### ex05_mbarrier.cu

**PASS** · Compile UTC: `2026-11-18T11:19:42.375062+00:00` · Executed UTC: `2026-11-18T11:19:43.203635+00:00`.

Source SHA-256: `c6a18427946c38f4305dc305ece5ade0461dffe09bce0fae6c87f3d53eefe046`.

[Exact source](validation/2026-11-18-two-h100/09-kernel-design-part-2/ex05_mbarrier/ex05_mbarrier.cu) · [Compile log](validation/2026-11-18-two-h100/09-kernel-design-part-2/ex05_mbarrier/compile.log) · [Run log](validation/2026-11-18-two-h100/09-kernel-design-part-2/ex05_mbarrier/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex05_mbarrier.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════╗
║  mbarrier Two-Barrier Handshake Demo                  ║
║  3 rounds of producer→consumer using full/empty bars  ║
╚══════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)

Producer values: [42.00, 99.50, 7.25]
Expected consumer outputs: [84.00, 199.00, 14.50]

Handshake Results:
  Round    Produced     Expected     Got          OK?
  ────────────────────────────────────────────────
  0        42.00        84.00        84.00        ✓ PASS
  1        99.50        199.00       199.00       ✓ PASS
  2        7.25         14.50        14.50        ✓ PASS

╔══════════════════════════════════════════════════════╗
║  ✓ ALL 3 ROUNDS PASSED                               ║
║                                                      ║
║  WHAT HAPPENED:                                      ║
║  1. empty_bar pre-signaled → producer starts freely  ║
║  2. Producer writes SMEM → arrives on full_bar       ║
║  3. Consumer waits full_bar → reads SMEM → doubles   ║
║  4. Consumer arrives on empty_bar → stage recyclable  ║
║  5. phase bits prevent stale barrier reads           ║
╚══════════════════════════════════════════════════════╝
```

### ex06_pingpong.cu

**PASS** · Compile UTC: `2026-11-18T11:19:43.758040+00:00` · Executed UTC: `2026-11-18T11:19:44.640676+00:00`.

Source SHA-256: `d1b6baf573465e1a90bef7d3a51c91508fbe818dbb311a49c8fe532beb3930cd`.

[Exact source](validation/2026-11-18-two-h100/09-kernel-design-part-2/ex06_pingpong/ex06_pingpong.cu) · [Compile log](validation/2026-11-18-two-h100/09-kernel-design-part-2/ex06_pingpong/compile.log) · [Run log](validation/2026-11-18-two-h100/09-kernel-design-part-2/ex06_pingpong/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex06_pingpong.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════╗
║  Ping-Pong Pipeline Simulation                        ║
║  6 tiles, C0=even, C1=odd, K_TILES=4 per tile      ║
╚══════════════════════════════════════════════════════╝

Tile Assignment and Output Verification:
  MMA formula: accumulate k*100+tile_id for k=0..4-1
  sum per tile = sum(k=0..3 of k*100) + tile_id × 4
  = 600 + tile_id × 4
  After epilogue (×2):

  Tile   Consumer   Expected     Got          OK?
  ──────────────────────────────────────────────
  0      C0         1200         1200         ✓ PASS
  1      C1         1208         1208         ✓ PASS
  2      C0         1216         1216         ✓ PASS
  3      C1         1224         1224         ✓ PASS
  4      C0         1232         1232         ✓ PASS
  5      C1         1240         1240         ✓ PASS

Ping-Pong Behavior Check:
  Even tiles processed by Consumer 0: ✓ ALL CORRECT
  Odd tiles processed by Consumer 1:  ✓ ALL CORRECT

╔══════════════════════════════════════════════════════╗
║  ✓ ALL TESTS PASSED                                  ║
║                                                      ║
║  In production, C0's epilogue overlaps C1's MMA:     ║
║  C0: [MMA T0][EPI T0][MMA T2][EPI T2]               ║
║  C1:    [MMA T1][EPI T1][MMA T3][EPI T3]            ║
║  Tensor cores: NEVER idle (perfect overlap)          ║
╚══════════════════════════════════════════════════════╝
```

### ex07_epilogue.cu

**PASS** · Compile UTC: `2026-11-18T11:19:45.193648+00:00` · Executed UTC: `2026-11-18T11:19:46.042302+00:00`.

Source SHA-256: `bfdc7b7c4d6b42fb42aa5ee46b9ca7611ca2b0ba6be64c12bbdb05a7baff5bff`.

[Exact source](validation/2026-11-18-two-h100/09-kernel-design-part-2/ex07_epilogue/ex07_epilogue.cu) · [Compile log](validation/2026-11-18-two-h100/09-kernel-design-part-2/ex07_epilogue/compile.log) · [Run log](validation/2026-11-18-two-h100/09-kernel-design-part-2/ex07_epilogue/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex07_epilogue.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════╗
║  Epilogue Fusion Demo                                 ║
║  D = ReLU( α×acc + β×C + bias[row] )                 ║
╚══════════════════════════════════════════════════════╝

Input tile: 64×64, alpha=1.0, beta=0.5
Bias: bias[m] = m * 0.1
Expected ReLU clamped to 0: 1965/4096 elements

Verification:
  Total elements: 4096
  Errors: 0
  Max absolute error: 0.000000 (threshold: 1e-4)

Sample outputs:
  D[0][0] = 0.0000 (expected 0.0000)
  D[2][5] = 0.0000 (expected 0.0000)
  D[3][2] = 0.0000 (expected 0.0000, should be 0 if pre-ReLU < 0)

FP8 amax reduction result: 44.6000
  (This is max|D[m][n]| across the tile, used for FP8 scaling)

╔══════════════════════════════════════════════════════╗
║  ✓ ALL 4096 ELEMENTS CORRECT — epilogue fusion works!   ║
╚══════════════════════════════════════════════════════╝
```

### ex08_comprehensive.cu

**PASS** · Compile UTC: `2026-11-18T11:19:46.577755+00:00` · Executed UTC: `2026-11-18T11:19:47.903815+00:00`.

Source SHA-256: `7c9011cfcbcf2cc91ff61610b8b55e4d3548700e8435178fa9967dff80fd0964`.

[Exact source](validation/2026-11-18-two-h100/09-kernel-design-part-2/ex08_comprehensive/ex08_comprehensive.cu) · [Compile log](validation/2026-11-18-two-h100/09-kernel-design-part-2/ex08_comprehensive/compile.log) · [Run log](validation/2026-11-18-two-h100/09-kernel-design-part-2/ex08_comprehensive/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex08_comprehensive.cu -o ./example
./example
```

**Captured output**

```text
╔══════════════════════════════════════════════════════════╗
║  ex08: Comprehensive — Persistent Warp-Spec GEMM+ReLU   ║
╚══════════════════════════════════════════════════════════╝

GPU: NVIDIA H100 80GB HBM3 (SM 9.0)

Problem: M=256, N=256, K=256
Tile:    64×64 (K-step=16, stages=3)
Tiles:   16 (tiles_m=4 × tiles_n=4)
AI:      42.7 FLOP/Byte (COMPUTE-BOUND ✓)

Launch: 1 CTA × 256 threads (WG0=producer, WG1=consumer)
Persistent loop: processes all 16 tiles sequentially.

Runtime: 0.981 ms
GFLOPS: 34.19

Verification:
  Total elements: 65536
  Errors: 0 (threshold: 1e-3)
  Max abs error: 0.000000
  Sample: C[0][0]=0.0000 ref=0.0000
  Sample: C[5][7]=0.0273 ref=0.0273

Design Summary:
  Kernel model:    Arithmetic intensity (AI=43 FLOP/Byte)
  Warp spec:       WG0=producer (loads), WG1=consumer (computes)
  Pipeline stages: 3 (circular buffer in shared memory)
  Scheduling:      Persistent (1 CTA, processes 16 tiles)
  Rasterization:   Swizzle=2 (M-dimension locality)
  Epilogue:        ReLU fused (register-level, zero extra HBM)
  Production TODO: Replace __syncthreads() with mbarrier,
                   replace manual loads with TMA,
                   replace st.global with TMA store,
                   add setmaxnreg for true WS benefit.

╔══════════════════════════════════════════════════════════╗
║  ✓ ALL 65536 ELEMENTS CORRECT                              ║
║                                                        ║
║  ALL LESSON 3 CONCEPTS DEMONSTRATED:                   ║
║  ✓ Roofline classification (compute-bound)             ║
║  ✓ Warp specialization (producer + consumer)           ║
║  ✓ Circular pipeline buffer (N_STAGES=3)              ║
║  ✓ Persistent scheduling (all tiles in one kernel)     ║
║  ✓ Swizzled rasterization (L2 locality)                ║
║  ✓ Fused epilogue (ReLU in registers)                  ║
╚══════════════════════════════════════════════════════════╝
```

### ex09_streamk_roles.cu

**PASS** · Compile UTC: `2026-11-18T11:19:48.473250+00:00` · Executed UTC: `2026-11-18T11:19:51.172405+00:00`.

Source SHA-256: `fd8d2d61ed5f5245d114196d678918a58f33401445342963fdaaafef32499fcf`.

[Exact source](validation/2026-11-18-two-h100/10-stream-k-scheduling/ex09_streamk_roles/ex09_streamk_roles.cu) · [Compile log](validation/2026-11-18-two-h100/10-stream-k-scheduling/ex09_streamk_roles/compile.log) · [Run log](validation/2026-11-18-two-h100/10-stream-k-scheduling/ex09_streamk_roles/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex09_streamk_roles.cu -o ./example
./example
```

**Captured output**

```text
═══════════════════════════════════════════════════════
  ex09_streamk_roles — Stream-K Split Role Demo
  3 output tiles × 90 K-tiles/tile = 270 total iters
  4 CTAs
═══════════════════════════════════════════════════════

Split assignment:
  Split 0: tile=0 K_idx= 0 k_tile_count=67 role=FIRST
  Split 1: tile=0 K_idx=67 k_tile_count=23 role=FINAL
  Split 2: tile=1 K_idx= 0 k_tile_count=44 role=FIRST
  Split 3: tile=1 K_idx=44 k_tile_count=46 role=FINAL
  Split 4: tile=2 K_idx= 0 k_tile_count=22 role=FIRST
  Split 5: tile=2 K_idx=22 k_tile_count=68 role=FINAL

Kernel output:
  CTA 0 → tile 0: FIRST SPLIT [0,67), partial=2278
  CTA 2 → tile 1: FIRST SPLIT [0,44), partial=4950
  CTA 4 → tile 2: FIRST SPLIT [0,22), partial=4213
  CTA 3 → tile 1: FINAL SPLIT [44,90), waiting for lock>=44
  CTA 5 → tile 2: FINAL SPLIT [22,90), waiting for lock>=22
  CTA 1 → tile 0: FINAL SPLIT [67,90), waiting for lock>=67
  CTA 3 → tile 1: FINAL SPLIT merged: prior=4950 + mine=7245 = 12195
  CTA 5 → tile 2: FINAL SPLIT merged: prior=4213 + mine=16082 = 20295
  CTA 1 → tile 0: FINAL SPLIT merged: prior=2278 + mine=1817 = 4095

Verification:
  Tile 0: got=4095  expected=4095  PASS ✓
  Tile 1: got=12195  expected=12195  PASS ✓
  Tile 2: got=20295  expected=20295  PASS ✓

ALL PASS ✓
```

### ex10_dependent_launch.cu

**PASS** · Compile UTC: `2026-11-18T11:19:51.829636+00:00` · Executed UTC: `2026-11-18T11:19:52.650956+00:00`.

Source SHA-256: `36b6313722fe43c219e825b2fe4c4e0b5cbb1630ceb02d3fe1624c2c242a7640`.

[Exact source](validation/2026-11-18-two-h100/10-stream-k-scheduling/ex10_dependent_launch/ex10_dependent_launch.cu) · [Compile log](validation/2026-11-18-two-h100/10-stream-k-scheduling/ex10_dependent_launch/compile.log) · [Run log](validation/2026-11-18-two-h100/10-stream-k-scheduling/ex10_dependent_launch/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex10_dependent_launch.cu -o ./example
./example
```

**Captured output**

```text
═══════════════════════════════════════════════════════
  ex10_dependent_launch — GDC Dependent Grid Protocol
  N=256 elements
═══════════════════════════════════════════════════════

Launching Kernel A (producer)...
Launching Kernel B (consumer) with programmatic serialization...

Verification (first 8 elements):
  [  0] A wrote    1 (exp    1) ✓  B result 1001 (exp 1001) ✓
  [  1] A wrote    4 (exp    4) ✓  B result 1004 (exp 1004) ✓
  [  2] A wrote    7 (exp    7) ✓  B result 1007 (exp 1007) ✓
  [  3] A wrote   10 (exp   10) ✓  B result 1010 (exp 1010) ✓
  [  4] A wrote   13 (exp   13) ✓  B result 1013 (exp 1013) ✓
  [  5] A wrote   16 (exp   16) ✓  B result 1016 (exp 1016) ✓
  [  6] A wrote   19 (exp   19) ✓  B result 1019 (exp 1019) ✓
  [  7] A wrote   22 (exp   22) ✓  B result 1022 (exp 1022) ✓
  ... (248 more elements)

ALL PASS ✓
```

### ex11_nvlink_bandwidth.cu

**PASS** · Compile UTC: `2026-11-18T11:19:53.204677+00:00` · Executed UTC: `2026-11-18T11:19:54.198165+00:00`.

Source SHA-256: `67e71f448fd51c1e130ef1c87abeece5d694e5546f6c32c837ca44db30b6c305`.

[Exact source](validation/2026-11-18-two-h100/11-multi-gpu-nccl/ex11_nvlink_bandwidth/ex11_nvlink_bandwidth.cu) · [Compile log](validation/2026-11-18-two-h100/11-multi-gpu-nccl/ex11_nvlink_bandwidth/compile.log) · [Run log](validation/2026-11-18-two-h100/11-multi-gpu-nccl/ex11_nvlink_bandwidth/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex11_nvlink_bandwidth.cu -o ./example
./example
```

**Captured output**

```text
GPUs detected: 2
GPU 0: NVIDIA H100 80GB HBM3, SM 9.0, 132 SMs
GPU 1: NVIDIA H100 80GB HBM3, SM 9.0, 132 SMs
GPU 0 can access GPU 1 memory: YES (does not identify NVLink)
Record topology with nvidia-smi topo -m. Payload: 134217728 bytes
  Staged 0->1 full-buffer verification: PASS (0 byte errors)
  Staged complete-transfer time: 4.860716 ms, payload 27.613 GB/s
  Peer API 0->1 full-buffer verification: PASS (0 byte errors)
  Peer API transfer time: 0.347011 ms, payload 386.782 GB/s (link unclassified)
  Kernel 1->0 full-buffer verification: PASS (0 byte errors)
  Kernel grid 1 blocks: 137.613831 ms, payload 0.975 GB/s
  Kernel 1->0 full-buffer verification: PASS (0 byte errors)
  Kernel grid 132 blocks: 1.070992 ms, payload 125.321 GB/s
Full-buffer checks: PASS
```

### ex12_nccl_allreduce.cu

**PASS** · Compile UTC: `2026-11-18T11:19:56.952819+00:00` · Executed UTC: `2026-11-18T11:19:58.343801+00:00`.

Source SHA-256: `f0cc87820667319cc733016feb646d72e31d4bf2dd26a20b8329545b7c062889`.

[Exact source](validation/2026-11-18-two-h100/11-multi-gpu-nccl/ex12_nccl_allreduce/ex12_nccl_allreduce.cu) · [Compile log](validation/2026-11-18-two-h100/11-multi-gpu-nccl/ex12_nccl_allreduce/compile.log) · [Run log](validation/2026-11-18-two-h100/11-multi-gpu-nccl/ex12_nccl_allreduce/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex12_nccl_allreduce.cu -o ./example -lnccl
./example
```

**Captured output**

```text
═══════════════════════════════════════════════════════
  ex12_nccl_allreduce — NCCL AllReduce End-to-End Demo
═══════════════════════════════════════════════════════

GPUs found: 2
  GPU 0: NVIDIA H100 80GB HBM3
  GPU 1: NVIDIA H100 80GB HBM3

  AllReduce results:
    N_GPUs:     2
    Data size:  33.55 MB (8M elements × float)
    Avg time:   0.149 ms
    Alg BW:     224.92 GB/s  (bytes / time)
    Bus BW:     224.92 GB/s  (2×(N-1)/N × alg BW)
    Expected result per element: 3
    First 8 values: 3 3 3 3 3 3 3 3
  GPU 1: PASS ✓ (result[0]=3, expected=3)
  GPU 0: PASS ✓ (result[0]=3, expected=3)

═══════════════════════════════════════════════════════
  Expected sum = 2 × (2 + 1) / 2 = 3
═══════════════════════════════════════════════════════
All-rank numerical verification: PASS
```

### ex13_hybrid_parallel_training.cu

**Compiled; eight GPUs required** · Compile UTC: `2026-11-18T11:19:59.944693+00:00` · Workload not launched on this two-GPU host.

Source SHA-256: `9a8dfd7c3904304b8cc1f344d6c91f4bb8748491a96cd6699455f9f012818cb2`.

[Exact source](validation/2026-11-18-two-h100/11-multi-gpu-nccl/ex13_hybrid_parallel_training/ex13_hybrid_parallel_training.cu) · [Compile log](validation/2026-11-18-two-h100/11-multi-gpu-nccl/ex13_hybrid_parallel_training/compile.log) · [Run log](validation/2026-11-18-two-h100/11-multi-gpu-nccl/ex13_hybrid_parallel_training/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex13_hybrid_parallel_training.cu -o ./example -lnccl -lcublas -Xcompiler -pthread
```

**Collector prerequisite diagnostic — workload not launched**

```text
Requires 8 visible GPU(s); discovered 2. Workload not executed.
```

## Additional numerical checks

[Attention output](validation/2026-11-18-two-h100/attention/run.log): **12 cases passed** against PyTorch 2.7.0 math SDPA, including partial tiles and causal masks. [Executed Python source](validation/2026-11-18-two-h100/attention/validate_attention.py).

[Nonuniform Chapter 7 manifest](validation/2026-11-18-two-h100/nonuniform/manifest.json) · [Captured output](validation/2026-11-18-two-h100/nonuniform/run.log). The variant checks 4,096 distinct exactly representable coordinates per tile; its device implementation is unchanged from the pinned archived baseline.

A successful sanitizer run checks the recorded memory accesses; it does not prove every asynchronous protocol or measure ordinary performance.
