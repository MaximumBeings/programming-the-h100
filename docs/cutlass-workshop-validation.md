# H100 CuTe/CUTLASS workshop execution — November 18, 2026

**Eight CUDA translation units compiled, two Python references byte-compiled, and all eleven workload invocations passed.** The GPU programs include a CuTe copy, a SIMT GEMM, two CUTLASS GEMMs and the serial/two-stream exercise. Three CUDA-compiled CuTe programs and both Python references execute CPU checks; they are not GPU-kernel benchmarks.

System: **1× NVIDIA H100 PCIe, compute capability 9.0, 81559 MiB**, driver **580.105.08**, CUDA compiler **12.8.93**, GCC **13.3.0**. CUTLASS **v4.3.6**, commit `1afc6d355b8cb992398f17beb4425702f1d57ddd`. All execution dates below are actual UTC stage timestamps.

Environment: **Ubuntu 24.04.5 LTS**, Linux **6.8.0-1064-nvidia**, Python **3.12.3**; Nsight Systems **2024.6.2.225**, Nsight Compute **2025.1.1.0**, Compute Sanitizer **2025.1.0.0**. [Captured OS](validation/2026-11-18-h100-workshop/system-9.log) · [Systems version](validation/2026-11-18-h100-workshop/system-4.log) · [Compute version](validation/2026-11-18-h100-workshop/system-5.log) · [Sanitizer version](validation/2026-11-18-h100-workshop/system-6.log).

[System metadata](validation/2026-11-18-h100-workshop/system.json) · [GPU inventory](validation/2026-11-18-h100-workshop/system-2.log) · [Results manifest](validation/2026-11-18-h100-workshop/results.json) · [Integrity inventory](validation/2026-11-18-h100-workshop/integrity.json).

The captured book checkout base and working-tree state are retained in metadata; each result is tied to its exact captured source hash. Earlier compilation/packaging failures and supplemental checks are recorded separately when linked below. These results do not establish B200 compatibility, native NVFP4, or an application speedup.

| Program | Scope | Actual run UTC | Result |
| --- | --- | --- | --- |
| [cute_layouts.cu](15-cute-layouts.md) | cute host oracle | `2026-11-18T14:09:25.346704+00:00` | PASS |
| [cute_algebra.cu](17-cute-layout-algebra.md) | cute host oracle | `2026-11-18T14:09:36.965003+00:00` | PASS |
| [cute_fragments.cu](21-mma-atoms.md) | cute host oracle | `2026-11-18T14:09:43.466571+00:00` | PASS |
| [cute_copy.cu](20-tiled-copy.md) | cuda workload | `2026-11-18T14:09:59.102270+00:00` | PASS |
| [tiled_gemm.cu](23-gemm-construction.md) | cuda workload | `2026-11-18T14:10:02.146468+00:00` | PASS |
| [cutlass_gemm.cu](24-cutlass-gemm.md) | cuda workload | `2026-11-18T14:10:34.405978+00:00` | PASS |
| [cutlass_gemm_relu.cu](26-epilogue-fusion.md) | cuda workload | `2026-11-18T14:11:16.060379+00:00` | PASS |
| [nsys_overlap.cu serial](14-nsight-systems.md) | cuda workload | `2026-11-18T14:11:19.004299+00:00` | PASS |
| [nsys_overlap.cu overlap](14-nsight-systems.md) | cuda workload | `2026-11-18T14:11:19.521997+00:00` | PASS |
| [indexing_workbook.py](19-cta-indexing.md) | python cpu reference | `2026-11-18T14:11:20.056089+00:00` | PASS |
| [fp4_reference.py](27-precision-block-scaling.md) | python cpu reference | `2026-11-18T14:11:20.188561+00:00` | PASS |

## Actual capture tools and reports

The two Nsight Systems captures contain the passing workload and a captured `transform` kernel summary. Capture completion is not evidence of useful overlap or speedup; inspect the timeline and compare the stated measurement boundary.

- **serial**: capture UTC `2026-11-18T14:11:20.222789+00:00`. [Capture log](validation/2026-11-18-h100-workshop/nsys-serial.log) · [Kernel summary](validation/2026-11-18-h100-workshop/nsys-serial-stats.log) · [Report](validation/2026-11-18-h100-workshop/timeline-serial.nsys-rep).
- **overlap**: capture UTC `2026-11-18T14:11:29.000569+00:00`. [Capture log](validation/2026-11-18-h100-workshop/nsys-overlap.log) · [Kernel summary](validation/2026-11-18-h100-workshop/nsys-overlap-stats.log) · [Report](validation/2026-11-18-h100-workshop/timeline-overlap.nsys-rep).

## Memory, race and counter checks

Four memory-check executions passed their numerical oracles and reported **zero errors**. The SIMT GEMM racecheck reported **zero hazards, zero errors and zero warnings**. These checks cover the captured executions, not every possible schedule.

| Check | Actual UTC | Captured result |
| --- | --- | --- |
| memcheck-cute_copy | `2026-11-18T14:14:45.726976+00:00` | [Actual output](validation/2026-11-18-h100-workshop/supplemental-captures/gpu-checks/20261118T141445Z/memcheck-cute_copy.log) |
| memcheck-tiled_gemm | `2026-11-18T14:14:46.596786+00:00` | [Actual output](validation/2026-11-18-h100-workshop/supplemental-captures/gpu-checks/20261118T141445Z/memcheck-tiled_gemm.log) |
| memcheck-cutlass_gemm | `2026-11-18T14:14:47.165149+00:00` | [Actual output](validation/2026-11-18-h100-workshop/supplemental-captures/gpu-checks/20261118T141445Z/memcheck-cutlass_gemm.log) |
| memcheck-cutlass_gemm_relu | `2026-11-18T14:14:47.833615+00:00` | [Actual output](validation/2026-11-18-h100-workshop/supplemental-captures/gpu-checks/20261118T141445Z/memcheck-cutlass_gemm_relu.log) |
| racecheck-tiled_gemm | `2026-11-18T14:14:48.502262+00:00` | [Actual output](validation/2026-11-18-h100-workshop/supplemental-captures/gpu-checks/20261118T141445Z/racecheck-tiled_gemm.log) |

A basic Nsight Compute capture of the selected `transform` kernel completed after a process-local privileged run. The original ordinary-user **ERR_NVGPUCTRPERM** diagnostic remains in the manifest/logs; no system-wide counter policy was changed. Counter replay changes timing, so the profiled application span is not an unprofiled performance result.

[Actual Compute report](validation/2026-11-18-h100-workshop/supplemental-captures/gpu-checks/20261118T141445Z/ncu-transform-privileged.ncu-rep) · [Captured counter details](validation/2026-11-18-h100-workshop/supplemental-captures/counter-analysis/20261118T141612Z/ncu-details.log).


## Additional introductory stride exercise

The separate [Chapter 16 stride oracle](16-cute-strides.md) was byte-compiled and executed locally on November 18, 2026. Its [commands and timestamps](validation/2026-11-18-strides/commands.json), [system information](validation/2026-11-18-strides/system.json) and [output](validation/2026-11-18-strides/run.log) identify a CPU Python address model, not another H100 workload. The ten-source H100 dataset above remains unchanged.

## Reproduce the run

Follow [the introductory workspace](15-cute-layouts.md) and [the complete test plan](examples/advanced/README.md). The collector requires a clean pinned CUTLASS dependency, a fresh output directory and explicit `--execute --profiles`. It never treats a host model as a numerical GPU workload.
