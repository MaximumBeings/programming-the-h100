# Part XII: Inference Systems on Hopper

!!! abstract "What this part is"

    Chapters 1 to 31 teach the machine: how a Hopper GPU moves and multiplies data. Part XII asks what happens when those kernels become a **serving system**: many users, long contexts, several GPUs, and a bill. Each chapter is one buildable project that ends in a number you can publish.

The project list comes from a public LinkedIn post by Ravi A Samy listing twelve inference-engineering portfolio projects. The descriptions, numbers and code in this book are original to it.

!!! warning "How to read the evidence in this part"

    Every program in Part XII was run on a **CPU** only. That validates logic (routing, accounting, queueing, correctness of a communication pattern) and arithmetic (byte counts, bandwidth floors). It does **not** validate GPU speed, NCCL or InfiniBand throughput, or real-model quality. Each chapter separates "checked on CPU" from "untested on H100", and where a number is a datasheet value rather than a measurement it says so. Results from a real H100 cluster should replace the simulated tables.

## The twelve projects and where they live

| # | Project | Where |
|---|---|---|
| 1 | Custom Triton kernel optimizer | Triton from First Principles, Part 8 |
| 2 | Disaggregated prefill/decode cluster | [Chapter 35](35-disaggregated-prefill-decode.md) |
| 3 | Hierarchical KV-cache offloader | [Chapter 34](34-kv-cache-offloading.md) |
| 4 | Grammar-constrained decoding engine | Triton from First Principles, Part 8 |
| 5 | MoE expert-parallel router | [Chapter 33](33-moe-expert-parallel.md) |
| 6 | FP4/INT4 quantization and calibration pipeline | Triton from First Principles, Part 8; native FP4 in the Blackwell appendix (planned) |
| 7 | Speculative decoding harness | Triton Part 8 (verify kernel); [Chapter 37](37-speculative-decoding-70b.md) (70B cost model) |
| 8 | Queue-based GPU autoscaler | Part XIII (planned) |
| 9 | Semantic AI gateway and cache | Part XIII (planned) |
| 10 | Multi-model complexity router | Part XIII (planned) |
| 11 | WebGPU / MLX edge fallback | Part XIII (planned) |
| 12 | Public inference benchmark teardown | [Chapter 36](36-benchmark-teardown.md) |

The Triton book is at <https://maximumbeings.github.io/triton-from-first-principles/>.

## The shared harness

Every project is judged with the same instruments, introduced in [Chapter 32](32-inference-engine-and-harness.md): a fixed workload with Poisson arrivals, time to first token (TTFT), time per output token (TPOT), percentile latencies, goodput under a service-level objective (SLO), and dollars per million tokens. Report a number only alongside the workload, software versions and hardware that produced it.
