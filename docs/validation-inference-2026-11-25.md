# Inference-system validation — November 25, 2026

The [inference-engine path](inference-systems-overview.md) now has fresh reference runs and a working [CUDA/vLLM lab](47-vllm-tuning-lab.md). All final test commands below completed successfully. The two failed server startups are preserved, with their setup corrections explained below.

!!! note "Follow-up adversarial review"
    The current Python listings include later CPU corrections from the [agent audit](inference-adversarial-review-2026-11-25.md). That review has separate fresh CPU captures. This original dataset remains unchanged and proves only its archived source revisions. The shared HTTP client changed after this original real-vLLM capture; its current version now has both fresh CPU/mock validation and a separate [post-review real-vLLM rerun](validation-client-rerun-2026-11-25.md). No CUDA kernel source changed in that review.

## System and source identity

| Item | Captured value |
|---|---|
| GPU | 1× NVIDIA H100 80GB HBM3, 81,559 MiB reported |
| Driver | 595.112.07 |
| OS / CPU architecture | Ubuntu 24.04 userspace; Linux 6.8.0-149-generic; x86-64 |
| Python | 3.12.3 |
| Pod image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Reference/CUDA-probe PyTorch | 2.8.0+cu128, CUDA runtime 12.8 |
| Isolated serving stack | vLLM 0.11.2, PyTorch 2.9.0+cu128, Triton 3.5.0 |
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, BF16 |
| Resolved model revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Baseline checkout | `1a869514e1bdbf3426d7467d6264b26758b74250` |
| Date | November 25, 2026 (UTC); collection timestamp in the inventory |

The toolkit maximum printed by `nvidia-smi` is driver capability, not the CUDA runtime version of either PyTorch wheel. The two Python environments are intentionally recorded separately.

Raw [system information](validation/2026-11-25-inference/system.json), [commands](validation/2026-11-25-inference/commands.json), [dependency lock](validation/2026-11-25-inference/serving-requirements.lock.txt), [pip check](validation/2026-11-25-inference/pip-check.log) and [SHA-256 inventory](validation/2026-11-25-inference/checksums.json) are retained. Source snapshots are stored under `sources-baseline` and `sources-current` within this dataset. The checksums bind these local records; they are not independent third-party attestation of execution.

## What was executed

| Run family | Result | What it establishes |
|---|---|---|
| Chapters 32–46 plus FP4 format reference | 16 scripts, 30 invocations, all exit 0 | CPU reference tests and full demonstrations reran. These are not GPU performance measurements. |
| Corrected Chapter 33 placement | Full program exit 0; boundary regression passes | Unsupported equal-slot counts are rejected; default routing/dispatch examples still pass on CPU/Gloo. |
| Corrected Chapter 43 allocator/cache | Full program exit 0; boundary regression passes | Invalid K/V shape cannot leak blocks or mutate fork ownership; negative indices are rejected. |
| Corrected Chapter 36 client | Self-tests and complete mock sweep exit 0 | Malformed streams fail, single-token TPOT is zero, existing measurement demonstrations still run. |
| CUDA paged-attention probe | 12 numerical comparisons pass; empty input rejected | FP32/FP16/BF16 inputs and lengths 1/3/17/65 match a float64 dense reference after input quantization. |
| Real vLLM completions | Repeated greedy outputs match | The installed model and non-streaming endpoint execute. This is not a semantic-quality test. |
| Real vLLM streaming | Warm-up plus 8 measured streams pass; each measured stream has 32 output tokens | Current streaming client executes against a real model on CUDA. |

The [reference runner](https://github.com/MaximumBeings/programming-the-h100/blob/main/tools/run_inference_references.py) is available in the repository; run it with `--output /tmp/inference-reference-rerun` after installing NumPy and PyTorch. Chapter 47 requires its own explicit CUDA and server commands. The [baseline result index](validation/2026-11-25-inference/references/results.json) contains source hashes, commands, durations and per-run logs. The corrected program logs are [Chapter 33](validation/2026-11-25-inference/corrected/ch33_moe_expert_parallel.log), [Chapter 43](validation/2026-11-25-inference/corrected/ch43_paged_kv_cache.log), [Chapter 36 full sweep](validation/2026-11-25-inference/corrected/ch36-full.log) and [new boundary regressions](validation/2026-11-25-inference/corrected/boundaries-final.log).

## CUDA numerical sample

The beginning of the [raw CUDA log](validation/2026-11-25-inference/paged-cuda.log), reproduced verbatim:

```text
PASS torch.float32 length=1 max_abs_error=0
PASS torch.float32 length=3 max_abs_error=1.37e-07
PASS torch.float32 length=17 max_abs_error=1.04e-07
PASS torch.float32 length=65 max_abs_error=9.28e-08
PASS torch.float16 length=1 max_abs_error=0
PASS torch.float16 length=3 max_abs_error=1.71e-07
PASS torch.float16 length=17 max_abs_error=1.07e-07
PASS torch.float16 length=65 max_abs_error=7.77e-08
PASS torch.bfloat16 length=1 max_abs_error=0
PASS torch.bfloat16 length=3 max_abs_error=1.04e-07
PASS torch.bfloat16 length=17 max_abs_error=8.07e-08
PASS torch.bfloat16 length=65 max_abs_error=5.64e-08
PASS empty-visible-key input rejected
```

The implementation uses a sequence of PyTorch CUDA operations with FP32 accumulation. It is not a fused custom PagedAttention kernel, and these correctness cases do not establish its throughput.

## Real serving sample

The [probe JSON](validation/2026-11-25-inference/probe.json) retains all eight individual records and the [probe stdout](validation/2026-11-25-inference/probe.log) retains the summary and generated text. Both repeated completions reported 170 prompt tokens and 32 completion tokens. The text below is an intentionally truncated 32-token raw completion, without a chat template. It illustrates execution, not answer quality:

```text
1. **Understanding the Problem:**
   - We are dealing with an Autoregressive (AR) model, which is a type of time series model where
```

Rounded summaries of this short **serial smoke test**:

| Metric | Observed value |
|---|---|
| Measured streams / failures | 8 / 0 |
| TTFT p50 | 13.58 ms |
| Average-per-request TPOT p50 | 6.72 ms |
| Output tokens / elapsed measurement window | 141.97 tokens/s |

Eight serial requests are insufficient for a reliable tail-latency or capacity claim. This is not a comparison with SGLang or TensorRT-LLM, not a 70B run, and not evidence for any of the earlier simulated speedups. The [raw server metrics](validation/2026-11-25-inference/metrics.txt) and [successful startup/server log](validation/2026-11-25-inference/server-final.log) are separate from the client timings.

## Failures encountered and fixed before the successful run

1. **Mixed Python environments:** the initial `--system-site-packages` environment inherited the pod's existing PyTorch dependencies while installing a different PyTorch. Startup failed with `libcusparseLt.so.0` missing. The [first failure log](validation/2026-11-25-inference/server-first-failure.log) remains exact. Disabling system-site inheritance and installing the complete dependency set fixed this; the [isolated install log](validation/2026-11-25-inference/install-isolated.log) and clean `pip check` are retained. Chapter 47 starts with an isolated venv.
2. **Inherited optional downloader flag:** the isolated server then found `HF_HUB_ENABLE_HF_TRANSFER=1` but no `hf_transfer` package. The [second failure log](validation/2026-11-25-inference/server-isolated.log) is preserved. Setting `HF_HUB_ENABLE_HF_TRANSFER=0` enabled the ordinary download path, after which the model loaded and all serving requests passed.

No raw log was rewritten to conceal either failure. The successful final run has separate files.

## Scope and next tests

This validates the listed programs and the small real-model serving path. The CPU mini engine still executes NumPy operations sequence by sequence; placing it on an H100 host does not make it a GPU engine. Its reported scheduler times remain simulated. The next engine work is batched/fused kernels in the Triton book and distributed ownership/collectives in the multi-GPU book.

The following still require their own hardware/workload validation: real multi-GPU MoE and RDMA disaggregation; 70B speculative decoding; actual CPU/NVMe KV offload performance; KEDA on a cluster; browser/MLX fallback; real semantic-cache and router quality; and a repeated open-loop multi-engine load sweep. At the time of this original capture, its listed fixes had completed their stated reruns. The follow-up agent audit records its additional source revisions and CPU tests separately; it does not relabel this historical serving capture as a run of the later shared client. These broader deployments are not claimed to have been tested by this session.

The temporary pod was stopped after the logs were exported and their archive checksum verified locally. Earlier execution datasets remain unchanged.
