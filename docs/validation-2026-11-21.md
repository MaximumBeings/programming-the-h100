# H100 revalidation — November 21, 2026

## System and scope

Two NVIDIA H100 80GB HBM3 GPUs, Ubuntu 24.04, x86-64, driver 580.105.08,
CUDA compiler 12.8.93, Python 3.12.3. Sources came from commit
`5461d2aa4bb763c9a2716a3a9d7c4e4184b0222c`.
The complete [system capture](validation/2026-11-21-two-h100/h100-supplement-20261121/system.log)
includes GPU UUIDs, topology and installed packages. Raw captures and archived
sources remain unchanged; a successful run is scoped to the recorded cases.

All 27 original complete programs compiled for `sm_90a`. Twenty-four executed
with one visible GPU; NVLink and NCCL were then rerun with both GPUs and passed.
Thus **26 current programs executed successfully**. `ex13_hybrid_parallel_training.cu`
requires eight GPUs and was not executed. Its compile success is not a workload pass.
The separate Chapter 30 attention reference passed seven complete O/LSE cases.
No CUTLASS FMHA rebuild, eight-GPU workload or new performance comparison is claimed here.

## Commands, dates and results

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/validate_examples.py --out /tmp/h100-rerun-20261121
python3 tools/validate_examples.py ex11_nvlink_bandwidth ex12_nccl_allreduce --out /tmp/h100-two-gpu-20261121
nvcc -std=c++17 -O2 -arch=sm_90a docs/examples/attention/attention_online.cu -o attention
CUDA_VISIBLE_DEVICES=1 ./attention
```

The first runner correctly returned a hardware-limitation status for its three
multi-GPU cases. The second runner returned zero after executing the two that
fit this host. Individual commands, UTC timestamps, source hashes and exit codes:
[first 27 records](validation/2026-11-21-two-h100/h100-rerun-20261121/results.json),
[two-GPU rerun](validation/2026-11-21-two-h100/h100-two-gpu-20261121/results.json),
[sanitizer manifest](validation/2026-11-21-two-h100/h100-supplement-20261121/manifest.json).
The separate initial attention compile/run logs do not contain per-command
timestamps; its later dated memcheck capture records a successful execution of the same binary.

### NVLink full-buffer check

```text
GPU 1: NVIDIA H100 80GB HBM3, SM 9.0, 132 SMs
GPU 0 can access GPU 1 memory: YES (does not identify NVLink)
Record topology with nvidia-smi topo -m. Payload: 134217728 bytes
  Staged 0->1 full-buffer verification: PASS (0 byte errors)
  Staged complete-transfer time: 4.853771 ms, payload 27.652 GB/s
  Peer API 0->1 full-buffer verification: PASS (0 byte errors)
  Peer API transfer time: 0.346973 ms, payload 386.825 GB/s (link unclassified)
  Kernel 1->0 full-buffer verification: PASS (0 byte errors)
  Kernel grid 1 blocks: 137.840863 ms, payload 0.974 GB/s
  Kernel 1->0 full-buffer verification: PASS (0 byte errors)
  Kernel grid 132 blocks: 1.070531 ms, payload 125.375 GB/s
Full-buffer checks: PASS
```

[Full output](validation/2026-11-21-two-h100/h100-two-gpu-20261121/11-multi-gpu-nccl/ex11_nvlink_bandwidth/run.log).

### NCCL all-rank check

```text
    Avg time:   0.265 ms
    Alg BW:     126.86 GB/s  (bytes / time)
    Bus BW:     126.86 GB/s  (2×(N-1)/N × alg BW)
    Expected result per element: 3
    First 8 values: 3 3 3 3 3 3 3 3 
  GPU 1: PASS ✓ (result[0]=3, expected=3)
  GPU 0: PASS ✓ (result[0]=3, expected=3)

═══════════════════════════════════════════════════════
  Expected sum = 2 × (2 + 1) / 2 = 3
═══════════════════════════════════════════════════════
All-rank numerical verification: PASS
```

[Full output](validation/2026-11-21-two-h100/h100-two-gpu-20261121/11-multi-gpu-nccl/ex12_nccl_allreduce/run.log).

### Attention reference

```text
PASS worked-full: all 4 O values and 2 LSE values checked; first O=[1.6, 2]
PASS worked-causal: all 4 O values and 2 LSE values checked; first O=[1, 0]
PASS worked-offset-1: all 4 O values and 2 LSE values checked; first O=[0.6666667, 0.6666667]
PASS worked-reordered: all 4 O values and 2 LSE values checked; first O=[1.6, 2]
PASS rectangular-full: all 85 O values and 17 LSE values checked; first O=[0.001319525, -0.002025958]
PASS rectangular-causal: all 85 O values and 17 LSE values checked; first O=[-0.6875, -0.375]
PASS rectangular-offset-1: all 85 O values and 17 LSE values checked; first O=[-0.6289223, -0.3164222]
CUDA ATTENTION PASS: seven cases; complete O/LSE checks; no performance claim.
```

## Synchronization and memory checks

`ex01_cluster_inspect`, `ex02_ptx_basics`, `ex06_comprehensive` and
`ex02_reduce_async` each passed Compute Sanitizer memcheck, racecheck and
synccheck with `--error-exitcode 91`. The Chapter 30 attention reference also
passed memcheck: **13 successful sanitizer invocations**. Exact commands,
dates, zero exits and clean summaries are preserved in the sanitizer manifest.
These tools cover the executed paths; they are not proofs of all synchronization protocols.

`python tools/verify_current_h100.py` checks current source hashes against the
new archived sources, all captured-file hashes, the 26 successful program records,
the eight-GPU exclusion and clean sanitizer summaries. Historical logs from earlier
hosts remain intact and continue to describe their historical source revisions.
