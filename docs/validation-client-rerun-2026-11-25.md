# Corrected HTTP client: real-vLLM rerun — November 25, 2026

The remaining post-audit integration check is complete. The **exact current Chapter 36 client and Chapter 47 probe** ran against vLLM on a RunPod H100. All positive requests succeeded, and both deliberately induced failure cases were handled correctly. The pod was stopped after the evidence was exported and checksum-verified locally.

The client source matches commit `a9ed5561a518a5f646686cad6610f597721e90f9`. No client or kernel source was changed for this rerun. This record supplements the [earlier H100 capture](validation-inference-2026-11-25.md) and [agent-audit CPU reruns](inference-adversarial-review-2026-11-25.md); their raw files remain unchanged.

## System and execution identity

| Item | Recorded value |
|---|---|
| Execution date/time | November 25, 2026, 10:49:59–10:50:09 AM EST |
| GPU | NVIDIA H100 80GB HBM3; 132 SMs; 81,559 MiB reported |
| Driver | 595.112.07 |
| OS / architecture | Ubuntu 22.04.5 LTS; Linux 6.8.0-149-generic; x86-64 |
| Python | 3.12.12 |
| Serving image | `vllm/vllm-openai:v0.11.2` |
| Image digest | `sha256:2c908d5a84ed251b6a17d179f42d06df1aff353007779ac5eecd8a0ea3fe9331` |
| vLLM / PyTorch | 0.11.2 / 2.9.0+cu129 |
| PyTorch CUDA runtime / NumPy | 12.9 / 2.2.0 |
| Model and tokenizer revision | Qwen/Qwen2.5-0.5B-Instruct; `7ae557604adf67be50417f59c2c2f167def9a775` |
| Execution mode | BF16; single GPU; eager mode; prefix caching and chunked prefill enabled |

The image's CUDA 12.9 runtime differs from the earlier pip-installed CUDA 12.8 stack. Both environments are recorded separately. The small timing samples are execution observations, not evidence of a performance change between environments. Command timestamps are retained in UTC; raw server log timestamps are preserved as emitted.

Raw [system information](validation/2026-11-25-client-rerun/system.json), [OS information](validation/2026-11-25-client-rerun/os-release.txt), [dependency lock](validation/2026-11-25-client-rerun/requirements.lock.txt), [commands and exit codes](validation/2026-11-25-client-rerun/commands.json), [server log](validation/2026-11-25-client-rerun/server.log) and [checksum inventory](validation/2026-11-25-client-rerun/checksums.json) are retained. The [client snapshot](validation/2026-11-25-client-rerun/sources/ch36_benchmark_teardown.py) and [probe snapshot](validation/2026-11-25-client-rerun/sources/ch47_vllm_probe.py) match the current runnable downloads byte for byte.

## What passed

| Test | Outcome |
|---|---|
| Two repeated non-streaming greedy completions | Both returned 32 output tokens; text matched exactly. |
| Streaming warm-up | Completed successfully; excluded from measured serial records. |
| Eight serial streaming requests | 8/8 succeeded; every reply reported exactly 32 output tokens. |
| Open-loop CLI run | 16/16 requests succeeded at an offered 2 requests/s; the first two are warm-up, leaving 14 in the summary. |
| Unknown model | Expected HTTP 404 was recorded as a failure with a finite terminal timestamp. |
| Deliberately short client deadline | Expected timeout was recorded as a failure with a finite terminal timestamp. |

The two negative cases are **expected failures of requests and successful checks of the client's error handling**. They are not failed model-correctness tests. Their undefined first-token timestamps are stored as JSON `null`; the normalization is visible in the [negative-check source](validation/2026-11-25-client-rerun/sources/real_client_negative_checks.py).

All three test commands exited zero. The [suite log](validation/2026-11-25-client-rerun/suite.log) ends:

```text
probe 0
open_loop 0
negative 0
ALL REAL CLIENT CHECKS PASS
```

## Captured sample outputs

Rounded client summaries:

| Metric | Eight serial streams | Open-loop summary after two warm-up requests |
|---|---|---|
| Reported request count | 8 | 14 |
| Unexpected failures | 0 | 0 |
| TTFT p50 | 16.80 ms | 21.26 ms |
| TPOT p50 | 7.39 ms | 7.21 ms |
| Output tokens / measurement window | 129.40 tokens/s | 44.36 tokens/s |

These are short smoke/integration runs. The open-loop test is below a saturation study and the sample sizes do not support a reliable p99, capacity limit or multi-engine performance comparison. Repeated greedy equality checks reproducibility in this run, not semantic answer quality or universal determinism.

The exact [probe stdout](validation/2026-11-25-client-rerun/probe.log) and [probe JSON](validation/2026-11-25-client-rerun/probe.json) include the raw model text, usage counts and eight timing records. The [open-loop stdout](validation/2026-11-25-client-rerun/open_loop.log) and [all 16 request records](validation/2026-11-25-client-rerun/open-loop.jsonl) are separate files. The publication verifier recomputes the open-loop summary from those records and compares it with the captured stdout.

The [negative-case capture](validation/2026-11-25-client-rerun/negative.json) records the two expected errors:

```text
unknown_model: HTTP/1.1 404 Not Found
client_deadline: timeout
```

Both have `ok=false` and `end >= intended`, confirming that real-server failure handling retains terminal timing. The [raw negative stdout](validation/2026-11-25-client-rerun/negative.log) preserves the complete records.

## Reproduce the integration checks

Use the [Chapter 47 serving instructions](47-vllm-tuning-lab.md). This capture uses the official `vllm/vllm-openai:v0.11.2` image and pins both revisions:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --tokenizer-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 2048 --max-num-seqs 16 \
  --max-num-batched-tokens 512 --gpu-memory-utilization 0.25 \
  --enable-prefix-caching --enable-chunked-prefill --enforce-eager --seed 42
```

With the server healthy, run from `docs/examples/inference-systems`:

```bash
python ch47_vllm_probe.py --model Qwen/Qwen2.5-0.5B-Instruct --output probe.json
python ch36_benchmark_teardown.py run \
  --url http://127.0.0.1:8000/v1/completions \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --rate 2 --n 16 --warmup 2 --seed 1 --n-in 64 --n-out 16 \
  --out open-loop.jsonl
```

The [captured execution helper](validation/2026-11-25-client-rerun/execute.py) and negative-check source show the readiness wait and expected-failure checks. The full source and command identities are bound to the capture inventory. The CI verifier checks records and source identity; it does not rent or execute a GPU.

## Remaining scope

The corrected client's real-vLLM integration item is closed. Larger model/load sweeps, multi-GPU/RDMA deployments, native B200 FP4 and real cache/router quality evaluations remain separate experiments. The earlier CPU simulation tables and earlier CUDA captures retain their own stated scope.
