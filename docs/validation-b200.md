# H100-book compatibility checks on eight B200 GPUs

**November 19, 2026.** This is a separate architecture-compatibility pass, not a replacement for the recorded H100 runs. The host supplied eight NVIDIA B200 GPUs, CC 10.0, driver 580.105.08 and NVCC 12.8.93 on Ubuntu 24.04.5.

The pass covered **41 source cases**: 27 original examples, eight workshop CUDA translation units (not all use CuTe or CUTLASS), four Python references and two attention sources. **36 exited successfully: seven CPU-only checks and 29 GPU-using programs.** The CPU checks were four Python scripts and three CuTe host enumerators. GPU use is not synonymous with a numerical kernel oracle. **Five cases did not execute their intended Hopper kernels:** two WGMMA builds were rejected for `sm_100` and their Hopper fallback binaries were incompatible; two Hopper CUTLASS GEMMs and example 88 exited at architecture prerequisites. No CUTLASS kernel ran in this compatibility set; `cute_copy` was its one CuTe GPU kernel. The native appendix below is a separate successful CUTLASS test set. Nsight workload runs here did not collect profiling traces.

The native B200 replacements are in the [Blackwell appendix](appendix-blackwell-b200.md): checked SM100 GEMM, fused ReLU, and a scoped forward-attention adaptation. They have their own evidence and are not counted as unchanged Hopper executions.

[Raw system probes](validation/2026-11-19-b200-compatibility/results/system.json) · [Raw collector results](validation/2026-11-19-b200-compatibility/results/results.json) · [Capture hashes](validation/2026-11-19-b200-compatibility/integrity.json). The original collector mistakenly labeled the upstream Hopper FMHA prerequisite exit as completed; the architecture message in its raw log establishes that no attention kernel ran. The table below corrects that interpretation. Raw logs are unchanged.

## What actually ran

The CuTe layout/algebra/fragment programs and the four Python references check host arithmetic. The remaining completed sources have their stated scope, including the eight-GPU hybrid-training example, TMA/barrier examples, checked transfers and the simple attention reference. Each source was rebuilt for the applicable B200 target; generic `sm_100` is distinct from `sm_100a`.

| Source case | Final interpretation | Captured execution UTC | Log |
| --- | --- | --- | --- |
| ex01_cluster_inspect.cu | Completed (source-specific scope) | `2026-11-19T22:04:13.895117+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex01_cluster_inspect/run-0.log) |
| ex02_ptx_basics.cu | Completed (source-specific scope) | `2026-11-19T22:04:12.692525+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex02_ptx_basics/run-0.log) |
| ex03_state_spaces.cu | Completed (source-specific scope) | `2026-11-19T22:04:12.592617+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex03_state_spaces/run-0.log) |
| ex04_ptx_pointers.cu | Completed (source-specific scope) | `2026-11-19T22:04:12.743555+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex04_ptx_pointers/run-0.log) |
| ex05_ptx_datatypes.cu | Completed (source-specific scope) | `2026-11-19T22:04:12.894404+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex05_ptx_datatypes/run-0.log) |
| ex06_comprehensive.cu | Completed (source-specific scope) | `2026-11-19T22:04:22.295924+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex06_comprehensive/run-0.log) |
| ex01_mbarrier_basic.cu | Completed (source-specific scope) | `2026-11-19T22:04:21.192258+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex01_mbarrier_basic/run-0.log) |
| ex02_war_prevention.cu | Completed (source-specific scope) | `2026-11-19T22:04:21.391099+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex02_war_prevention/run-0.log) |
| ex01_tma_basic.cu | Completed (source-specific scope) | `2026-11-19T22:04:21.543110+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex01_tma_basic/run-0.log) |
| ex02_tma_3d.cu | Completed (source-specific scope) | `2026-11-19T22:04:23.996770+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex02_tma_3d/run-0.log) |
| ex01_bulk_raw.cu | Completed (source-specific scope) | `2026-11-19T22:04:30.088908+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex01_bulk_raw/run-0.log) |
| ex02_reduce_async.cu | Completed (source-specific scope) | `2026-11-19T22:04:30.338774+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex02_reduce_async/run-0.log) |
| ex01_wgmma_basic.cu | WGMMA build rejected for sm_100; Hopper fallback did not execute | `2026-11-19T22:04:31.957439+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex01_wgmma_basic/run-0.log) |
| ex01_wgmma_pipeline.cu | WGMMA build rejected for sm_100; Hopper fallback did not execute | `2026-11-19T22:04:32.111730+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex01_wgmma_pipeline/run-0.log) |
| ex01_roofline.cu | Completed (source-specific scope) | `2026-11-19T22:04:32.394212+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex01_roofline/run-0.log) |
| ex02_warp_spec.cu | Completed (source-specific scope) | `2026-11-19T22:04:38.304203+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex02_warp_spec/run-0.log) |
| ex03_setmaxnreg.cu | Completed after sm_100a rebuild | `2026-11-19T22:05:24.336086+00:00` | [Output](validation/2026-11-19-b200-compatibility/setmaxnreg-blackwell/run.log) |
| ex04_circular_buffer.cu | Completed (source-specific scope) | `2026-11-19T22:04:39.039124+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex04_circular_buffer/run-0.log) |
| ex05_mbarrier.cu | Completed (source-specific scope) | `2026-11-19T22:04:39.089314+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex05_mbarrier/run-0.log) |
| ex06_pingpong.cu | Completed (source-specific scope) | `2026-11-19T22:04:41.495016+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex06_pingpong/run-0.log) |
| ex07_epilogue.cu | Completed (source-specific scope) | `2026-11-19T22:04:47.737556+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex07_epilogue/run-0.log) |
| ex08_comprehensive.cu | Completed (source-specific scope) | `2026-11-19T22:04:49.040477+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex08_comprehensive/run-0.log) |
| ex09_streamk_roles.cu | Completed (source-specific scope) | `2026-11-19T22:04:49.458051+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex09_streamk_roles/run-0.log) |
| ex10_dependent_launch.cu | Completed (source-specific scope) | `2026-11-19T22:04:48.288671+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex10_dependent_launch/run-0.log) |
| ex11_nvlink_bandwidth.cu | Historical exit 0; broken error macro, corrected source needs rerun | `2026-11-19T22:04:50.422374+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex11_nvlink_bandwidth/run-0.log) |
| ex12_nccl_allreduce.cu | Completed (source-specific scope) | `2026-11-19T22:04:57.390110+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex12_nccl_allreduce/run-0.log) |
| ex13_hybrid_parallel_training.cu | Completed (source-specific scope) | `2026-11-19T22:04:57.990164+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/ex13_hybrid_parallel_training/run-0.log) |
| cute_algebra.cu | Completed (source-specific scope) | `2026-11-19T22:05:01.199228+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/cute_algebra/run-0.log) |
| cute_copy.cu | Completed (source-specific scope) | `2026-11-19T22:05:02.719197+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/cute_copy/run-0.log) |
| cute_fragments.cu | Completed (source-specific scope) | `2026-11-19T22:05:03.434975+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/cute_fragments/run-0.log) |
| cute_layouts.cu | Completed (source-specific scope) | `2026-11-19T22:05:05.827441+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/cute_layouts/run-0.log) |
| cutlass_gemm.cu | Hopper-only architecture guard | `2026-11-19T22:05:16.235233+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/cutlass_gemm/run-0.log) |
| cutlass_gemm_relu.cu | Hopper-only architecture guard | `2026-11-19T22:05:18.086368+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/cutlass_gemm_relu/run-0.log) |
| nsys_overlap.cu | Completed (source-specific scope) | `2026-11-19T22:05:06.745606+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/nsys_overlap/run-0.log) |
| tiled_gemm.cu | Completed (source-specific scope) | `2026-11-19T22:05:07.196382+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/tiled_gemm/run-0.log) |
| attention_walkthrough.py | Completed (source-specific scope) | `2026-11-19T22:05:08.496531+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/attention_walkthrough/run-0.log) |
| fp4_reference.py | Completed (source-specific scope) | `2026-11-19T22:05:08.545264+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/fp4_reference/run-0.log) |
| indexing_workbook.py | Completed (source-specific scope) | `2026-11-19T22:05:08.593650+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/indexing_workbook/run-0.log) |
| stride_walkthrough.py | Completed (source-specific scope) | `2026-11-19T22:05:08.641584+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/stride_walkthrough/run-0.log) |
| attention_online.cu | Completed (source-specific scope) | `2026-11-19T22:05:10.091589+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/attention_online/run-0.log) |
| 88_hopper_fmha.cu | Hopper-only architecture guard | `2026-11-19T22:05:45.125587+00:00` | [Output](validation/2026-11-19-b200-compatibility/results/88_hopper_fmha/run-0.log) |

## Register resizing

The generic `sm_100` compilation rejected `setmaxnreg`. A Hopper `sm_90a` fallback compiled but could not JIT on B200. The same original source then compiled for `sm_100a` and passed both producer and consumer groups' 128-value checks. No register-resizing algorithm was changed.

- compile: UTC `2026-11-19T22:05:23.547207+00:00`; exit **0**.
- run: UTC `2026-11-19T22:05:24.336086+00:00`; exit **0**.

**Actual successful B200 output**

```text
╔══════════════════════════════════════════════════════╗
║  setmaxnreg Demo — Dynamic Register Resizing         ║
╚══════════════════════════════════════════════════════╝

GPU: NVIDIA B200 (SM 10.0)

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

## Architecture guards are not successful numerical runs

The two raw WGMMA examples require Hopper. The two Hopper CUTLASS GEMMs also check CC 9.0 before launching. Upstream example 88 prints its Hopper prerequisite and exits before attention work on B200—even though that exit is zero. Replacing a guard without changing its architecture-specific MMA/storage implementation is not a port. The appendix instead uses native SM100 collectives and a separate Blackwell attention implementation.

The original `nsys_overlap` source completed both serial and overlap modes. Its two raw workload logs are preserved; no profiling trace was captured during this short rental window.

## Limits of these records

All printed `-arch=sm_90a` book commands produce Hopper-specific GPU code that cannot run on B200. This pass rebuilt applicable sources for `sm_100`, using `sm_100a` for architecture-specific register resizing. Changing a flag alone does not port WGMMA. The setmaxnreg example checks value preservation, not measured register redistribution.

The source/hash/log inventories check internal consistency, not independent hardware attestation or proof that a log is genuine. Evidence tarballs and some ad hoc follow-up drivers were not committed. The committed command arrays preserve those actual invocations; the compatibility runner log lacks its final summary line, so use the per-command records and exits. The original WGMMA build diagnostics and fallback outcomes are distinct; the linked run outputs alone do not prove the compiler diagnostics.

The ex11 capture includes warning #549-D from the historical self-initializing macro. Its zero exit does not validate that error path. The four archived source copies are immutable evidence of that defect, not current recommended sources. See the [current corrections and completed H100 reruns](review-fixes-2026-11-21.md).
