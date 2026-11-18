# Exercises for Chapters 14–17

These new sources have a separate validation collector. The original 27-program records do not cover them. Run on a Hopper host after following Getting Started and Chapter 16; retain the complete output directory before shutting down the instance.

```bash
bash tools/fetch_cutlass.sh "$HOME/cutlass-v4.3.6"
python3 tools/run_advanced_validation.py \
  --cutlass "$HOME/cutlass-v4.3.6" \
  --output "$HOME/h100-advanced-$(date -u +%Y%m%dT%H%M%SZ)" \
  --execute --profiles
```

The layout exercise runs on the CPU; the GEMM and overlap exercises launch CUDA kernels. A successful profiler capture is separate from a performance improvement. The collector checks four workload results, saves exact source hashes and commands, and retains actual timestamps, system logs and profiler reports. New captured output will be published after the next hardware session; chapters currently show expected checks rather than invented transcripts.
