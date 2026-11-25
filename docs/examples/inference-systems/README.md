# Run the complete inference examples

The complete implementations are printed in Chapters 32–47 and the FP4 appendix. Download [the source bundle](inference-systems.zip) to get every local import in one directory. Its contents match the chapter listings and individual `.py` downloads.

## CPU reference environment

Use Python 3.12 in an isolated environment. From this directory:

```bash
python3.12 -m venv .venv-reference
. .venv-reference/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-reference.txt
python -m pip check
```

NumPy is used by Chapters 38–40 and 42–46. Chapter 33 uses PyTorch and four local Gloo processes; it does not require four GPUs. The other CPU references use the standard library. The fresh CPU audit uses NumPy 2.5.5 and PyTorch 2.8.0. NumPy 2.2.6 can emit false matrix-multiply status warnings on Apple M4/Accelerate; the independent finite-output regression is retained, and the newer reference environment avoids those warnings. The earlier Linux/H100 serving environment is recorded separately.

Linux users who only want CPU PyTorch can install NumPy from PyPI and the matching PyTorch CPU wheel from the official PyTorch CPU index instead. CUDA and vLLM are separate environments described in [Chapter 47](https://maximumbeings.github.io/programming-the-h100/47-vllm-tuning-lab/).

## Commands from the unpacked bundle directory

| Program | Focused checks | Full demonstration |
|---|---|---|
| 32: harness | `python ch32_inference_harness.py --test` | `python ch32_inference_harness.py` |
| 33: MoE dispatch logic | `python ch33_moe_expert_parallel.py --test` | `python ch33_moe_expert_parallel.py` |
| 34: offload model | `python ch34_kv_offload.py --test` | `python ch34_kv_offload.py` |
| 35: disaggregation model | `python ch35_disaggregation.py --test` | `python ch35_disaggregation.py` |
| 36: benchmark client | `python ch36_benchmark_teardown.py --selftest` | `python ch36_benchmark_teardown.py` |
| 37: speculation model | included in the full run | `python ch37_spec_decode_70b.py` |
| 38: semantic cache | `python ch38_semantic_cache.py --selftest` | `python ch38_semantic_cache.py` |
| 39: router | `python ch39_complexity_router.py --selftest` | `python ch39_complexity_router.py` |
| 40: autoscaler | `python ch40_gpu_autoscaler.py --selftest` | `python ch40_gpu_autoscaler.py` |
| 41: resilience | `python ch41_edge_fallback.py tests` | `python ch41_edge_fallback.py` |
| 42: gateway | `python ch42_gateway_capstone.py tests` | `python ch42_gateway_capstone.py` |
| 43: paged KV | `python ch43_paged_kv_cache.py tests` | `python ch43_paged_kv_cache.py` |
| 44: batching | `python ch44_continuous_batching.py tests` | `python ch44_continuous_batching.py` |
| 45: prefix cache | `python ch45_radix_prefix_cache.py tests` | `python ch45_radix_prefix_cache.py` |
| 46: mini engine | `python ch46_mini_engine.py tests` | `python ch46_mini_engine.py` |
| FP4 formats | included in the full run | `python fp4_formats.py` |

Chapters 32, 34 and 35 also accept `--extras`. Chapter 36's full mock-server demonstration takes several minutes. Later full simulator experiments can also take minutes. They use assumed costs, not GPU timings. The Chapter 39 calibration subcommand needs a separately supplied labelled dataset; a synthetic demo is not a quality evaluation of a trained model.

## Local imports are part of the program

- Chapter 35 imports Chapter 32.
- Chapter 42 imports Chapters 36 and 38–41.
- Chapter 46 imports Chapters 43–45.
- Chapter 47's CUDA probe imports Chapter 43; its HTTP probe imports Chapter 36.

Keep these files together, or run the original file paths from the repository root. Do not paste a dependent program into an otherwise empty notebook and expect its sibling imports to exist.

## GPU and real-server checks

The Chapter 47 CUDA program refuses to count a CPU fallback as a pass. Its HTTP probe requires a running server on `127.0.0.1:8000`. Follow [the chapter](https://maximumbeings.github.io/programming-the-h100/47-vllm-tuning-lab/) for the serving environment, startup settings and exact commands. No script in this bundle provisions or rents a GPU.

## Evidence and scope

The [November 25 H100 record](https://maximumbeings.github.io/programming-the-h100/validation-inference-2026-11-25/) preserves the original runs. The [follow-up agent audit](https://maximumbeings.github.io/programming-the-h100/inference-adversarial-review-2026-11-25/) has a separate dated CPU rerun record. The corrected HTTP client also passed its [separate real-vLLM rerun](https://maximumbeings.github.io/programming-the-h100/validation-client-rerun-2026-11-25/). Neither a CPU simulation nor a short real-model serving probe establishes production throughput, model quality or distributed deployment performance.
