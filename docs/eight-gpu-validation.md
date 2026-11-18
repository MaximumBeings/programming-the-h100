# Eight-GPU A100 training validation

In the dated execution below, the portable Chapter 11 TP=2 × PP=2 × DP=2 teaching program completed on **eight NVIDIA A100 GPUs**, compiled for **`sm_80`**. This is a separate architecture-specific supplement; the original one-H100 and final two-H100 datasets remain unchanged. It is not an eight-H100 run. Hopper-specific WGMMA and TMA examples are not validated by this A100 execution.

- **Run captured:** `2026-11-18T00:03:00.400131+00:00` (UTC).
- **Finished:** `2026-11-18T00:03:13.511354+00:00` (UTC).
- **Compiler / process exit:** `0` / `0`.
- **Program result:** `All-rank training validation: PASS`.
- **Executed source SHA-256:** `9a8dfd7c3904304b8cc1f344d6c91f4bb8748491a96cd6699455f9f012818cb2`.

The revised program uses distinct DP inputs and checks all 16,384 final weight values against an independent CPU serial reference, plus all 16 per-step replica loss values and four DP replica-pair comparisons. The reference follows the same DP averaging and per-TP-shard gradient-clipping semantics, with explicit numerical tolerances. This is a reference comparison of the teaching program, not a training-quality or throughput benchmark, cross-node RDMA test, or validation on B200. Loss is checked against the reference for each step; changing inputs do not imply a monotonic loss trajectory.

Download the [exact executed source](validation/2026-11-18-eight-a100/ex13_hybrid_parallel_training.cu), [manifest](validation/2026-11-18-eight-a100/manifest.json), [compiler log](validation/2026-11-18-eight-a100/compile.log), [raw run log](validation/2026-11-18-eight-a100/run.log), and [system/topology log](validation/2026-11-18-eight-a100/system-info.log). The complete raw log is authoritative; thread print order is run-specific.

The captured compiler command, with only the remote source and output paths shortened, is:

```bash
nvcc -std=c++17 -arch=sm_80 -O2 ex13_hybrid_parallel_training.cu -o ex13_hybrid_parallel_training -lnccl -Xcompiler -pthread -lcublas
./ex13_hybrid_parallel_training
```

The manifest preserves the exact original absolute-path commands and UTC stage timestamps. Verify the immutable hashes and current chapter-source match with:

```bash
python3 tools/verify_eight_gpu_audit.py
```

The [two-H100 final audit](final-audit.md) still records 26 PASS workloads and one eight-GPU prerequisite block for its own historical sources and host. This supplement supersedes that prerequisite limitation for the revised portable Chapter 11 program only.
