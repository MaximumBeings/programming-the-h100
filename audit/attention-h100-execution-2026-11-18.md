# Attention code and H100 execution — November 18, 2026 (America/New_York)

The user requested complete code in Chapter 30 and supplied a new two-H100 instance. Chapter 30 now displays its complete standalone CUDA reference and the complete Python oracle, each with a download and run commands. Chapter 31 retains the complete pinned CUTLASS source/header bundle. A reviewer independently checked the new CUDA reference's dimensions, ownership, masks, numerical criteria and agreement with the prose; no correction to the tested CUDA source was needed.

## Actual execution

The host had two NVIDIA H100 80GB HBM3 GPUs with 132 SMs, NV18 topology, Ubuntu 22.04.5 LTS, kernel 6.8.0-1064-nvidia, driver 580.105.08, CUDA compiler 12.8.93, GCC 11.4.0, CMake 3.22.1 and Python 3.10.12. Dates shown in local New York time fall on November 18; the raw UTC records fall on November 19. Both are stated explicitly in the chapters.

- **Standalone CUDA reference:** source SHA-256 `860d4976162dbee47edd0153d0eb8ff8089471d6ff8b7271653e6d8effad7b66`; NVCC C++17/O2/sm_90a build passed. Seven cases validated every O/LSE value against an independent binary64 reference from the actual FP32 inputs, plus closed-form checks for the worked cases. Noncausal, top-left/offset causal, reordered keys, rectangular dimensions and a partial block are included. Compute Sanitizer memcheck passed with zero errors.
- **Pinned CUTLASS FMHA:** v4.3.6 commit `1afc6d355b8cb992398f17beb4425702f1d57ddd`; FP16 target built through the official CMake recipe. Six forward and two backward invocations passed all 17 expected verified variant results. O/LSE and dQ/dK/dV checks used the pinned vendor references. D=64 full-forward and full-backward memchecks also passed with zero errors.
- **Python oracle:** byte-compiled and executed again on the GPU host; its earlier dated local record remains unchanged.

CUDA-visible device identities were recorded through the Driver API, confirming the selected UUIDs. The standalone reference used visible-device selection 1 and CUTLASS used selection 0. These were single-GPU tests on a two-GPU host, not distributed attention. All 6,711 CUTLASS compiler-input files matched the complete local pinned tree. Exact source snapshots, commands, stage times, stdout/stderr and hashes were saved locally before telling the user the instance could be shut down.

## Published evidence and scope

`docs/validation/2026-11-18-h100-attention` contains the original successful collector records, a source-file hash inventory, the standalone sources, supplemental device probes and memory-check records. The full CUTLASS library tree remains reproducible from the immutable pin rather than duplicating its 155MB in the website. The example's local kernel/reference headers are already distributed in the source ZIP.

`tools/verify_attention_h100.py` enforces source/download/listing identity, immutable log inventories, actual runtime arguments and verified variant rows, displayed dates/output and three clean memchecks. CI also builds/runs the shared standalone row arithmetic in host-only mode. That CI check is distinct from the recorded CUDA execution.

Current pending-execution statements were replaced with actual records in both chapters, the homepage and setup guide. Historical audit notes retain the scope of the earlier review rather than being rewritten as if those reviews executed CUDA.

The small correctness workloads do not establish peak H100 performance, arbitrary numerical distributions, FP8/BF16, modified GQA/MQA paths or all advertised backward head sizes. The online reference is a simple SIMT teaching implementation; the separate pinned CUTLASS lab uses TMA/WGMMA. Their declared input/precision and reference contracts remain explicit.
