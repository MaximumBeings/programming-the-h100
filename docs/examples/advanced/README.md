# Exercises for Chapters 14–29

[Actual November 18 H100 results](../../cutlass-workshop-validation.md) provide the source-specific commands, system information, dates and outputs. The original 27-program suite has separate historical records. This workshop uses eight CUDA translation units and two Python reference programs, with 11 workload invocations: the overlap program runs in both serial and two-stream modes. Three CuTe executables enumerate layouts/traits on the CPU; five programs launch CUDA workloads; both Python programs are CPU-only references.

Prepare the pinned dependency from Chapter 15, then run the collector from the book repository root:

```bash
export CUTLASS_ROOT="$HOME/cutlass-v4.3.6"
bash tools/fetch_cutlass.sh "$CUTLASS_ROOT"
python3 tools/run_advanced_validation.py \
  --cutlass "$CUTLASS_ROOT" \
  --output "$HOME/h100-workshop-$(date -u +%Y%m%dT%H%M%SZ)" \
  --execute --profiles
```

Omit both `--execute` and `--profiles` for a side-effect-free plan. A fresh output directory is required. The collector records exact source copies/hashes, the real toolchain and hardware, stage UTC timestamps, commands, exit codes, explicit numerical pass markers and optional Nsight Systems reports. Python byte-compilation is recorded as Python validation, not CUDA compilation. A profiler capture is accepted only with a passing workload and an actual captured `transform` kernel summary.

| Source | Main chapter | Scope |
| --- | --- | --- |
| [cute_layouts.cu](cute_layouts.cu) | 15 | Real CuTe host coordinate checks |
| [cute_algebra.cu](cute_algebra.cu) | 16 | Nested layouts, coalesce and composition |
| [cute_fragments.cu](cute_fragments.cu) | 20 | Actual pinned MMA trait ownership |
| [cute_copy.cu](cute_copy.cu) | 19 | Scalar CuTe GPU copy, predicates, ownership and guards |
| [tiled_gemm.cu](tiled_gemm.cu) | 22 | SIMT GPU GEMM, padding, batches, ragged dimensions and K=0 |
| [cutlass_gemm.cu](cutlass_gemm.cu) | 23 | Hopper Tensor Core GEMM with a full CPU oracle |
| [cutlass_gemm_relu.cu](cutlass_gemm_relu.cu) | 25 | Supported fused ReLU epilogue with a full CPU oracle |
| [nsys_overlap.cu](nsys_overlap.cu) | 14 | Serial/two-stream CUDA work and complete output checks |
| [indexing_workbook.py](indexing_workbook.py) | 18/28 | CPU arithmetic and modeled ownership/state checks |
| [fp4_reference.py](fp4_reference.py) | 26 | CPU E2M1 format/layout reference, not native NVFP4 execution |

Preserve the complete results directory before shutting down the instance. Expected checks in the prose explain the oracle; published execution blocks identify actual captured source revisions and outputs. A new source change needs its own rerun.

## Additional introductory stride oracle

[Chapter 16](../../16-cute-strides.md) adds [stride_walkthrough.py](stride_walkthrough.py), a separate CPU integer-address model. Run it with `python3 docs/examples/advanced/stride_walkthrough.py` from the repository root, without Python optimization flags that disable assertions. Its [dated local record](../../validation/2026-11-18-strides/commands.json) and [system information](../../validation/2026-11-18-strides/system.json) are separate from the frozen ten-source H100 collector above. It does not compile CuTe or execute a GPU kernel.
