# Independent review corrections — November 21, 2026

This repair pass addresses the November 19 and November 21 independent reviews.
The original execution logs and source snapshots are preserved. A historical
PASS describes its archived source, not a later corrected program.

## GPU revalidation pending

No GPU was available for this repair pass. Changed complete CUDA programs need
fresh compilation, numerical checks and applicable memory/race checks on Hopper.
The current source revisions and their archived predecessors are pinned in the
repair manifest. Until rerun, do not cite their historical outputs as validation
of the corrections.

The Chapter 2 fixes protect remote shared-memory readers before CTA exit and
remove the special-register output race. Chapter 5 adds an async-proxy fence and
alignment to the reduction source. Chapter 11 replaces the self-initializing
error macro. Other source edits correct explanatory comments or output labels;
they also change source identity and are not presented as newly GPU-tested.

## Historical evidence and scope

Four archived copies of the broken Chapter 11 macro remain intentionally
unchanged, documenting exactly what ran. Use the corrected current chapter
source for future runs. The old compiler warnings and zero exits do not validate
that error-checking path. The B200 compatibility page separates seven CPU-only
checks from 29 GPU-using programs and identifies five architecture exclusions.
Its native CUTLASS appendix has a distinct test set.

Evidence verifiers check hashes, commands, dates, inventories and captured
results. They do not independently attest hardware identity or rerun kernels.

## Exact pending source revisions

| Chapter / program | Archived SHA-256 | Current SHA-256 |
| --- | --- | --- |
| [ex01_cluster_inspect.cu](02-clusters-inline-ptx.md) | `ff9c3f47e5bc98d502ca02011672ee2fe024f52e93614d3eb91284953ce6e2f3` | `f4324866efbb03aa6ac27632000d031c87b1423a6f5755133aa96bbb8510389a` |
| [ex02_ptx_basics.cu](02-clusters-inline-ptx.md) | `50b26d6098753bb1cf202e4482308bf86020d3e0eee4fad0b59f2cfcc2eef70f` | `968130f4fabb8a50aa126c297105214ddfa15f03d554bf829266335c300cc426` |
| [ex06_comprehensive.cu](02-clusters-inline-ptx.md) | `4c5fb40a537b3c5dfc2190023302f953f98867349e486a5daeba6e73e823be8d` | `d0363f7713bdce71b4087b1d9dbc57141d586a8d109a3931a4a207ab8ab773ad` |
| [ex02_reduce_async.cu](05-cp-async-bulk-l2-hinting.md) | `52617e1a22a4619a1fd65dc92d4eac529b3bd02ee2ea34803ab51213f829f39a` | `dd125d7cb35b0c188bc09e9d471a366fc4321add59efa4210f2d5b82f5832132` |
| [ex01_wgmma_basic.cu](06-wgmma-part-1.md) | `282064c18eedf80395d5c5ed41bc0d1e4160c8d3886bf831c90b68628b84ad04` | `a557eb99705eca84366ee23ebd3bea87eadb25edbc69310e5ac9da9a7d3835e1` |
| [ex01_roofline.cu](08-kernel-design-part-1.md) | `d4e66adac175c64e13c8761c308a1a5c2151164a5505cd6959d04b8abab61449` | `4f3798b5519e14242c962b0f883327501d63e27f41d34afe2de840c67655761c` |
| [ex05_mbarrier.cu](09-kernel-design-part-2.md) | `c6a18427946c38f4305dc305ece5ade0461dffe09bce0fae6c87f3d53eefe046` | `ee8a728dbb60691c48a16525ccdb2fbdac87b40658ebb110d153efcb9d343edc` |
| [ex06_pingpong.cu](09-kernel-design-part-2.md) | `d1b6baf573465e1a90bef7d3a51c91508fbe818dbb311a49c8fe532beb3930cd` | `24cae60ab9d067e7337716ff196bf9ff722167c07ff6d05e946b66b20702dccf` |
| [ex08_comprehensive.cu](09-kernel-design-part-2.md) | `7c9011cfcbcf2cc91ff61610b8b55e4d3548700e8435178fa9967dff80fd0964` | `dc235caa169f07990a5ab75fbd6779feab3142180ed4f0d1e5675a4c4601508b` |
| [ex11_nvlink_bandwidth.cu](11-multi-gpu-nccl.md) | `67e71f448fd51c1e130ef1c87abeece5d694e5546f6c32c837ca44db30b6c305` | `e42e582e8510253e85a7eaf56e9af540493068b93fde85502c7a2026e63162d8` |
| [attention_online.cu](30-cutlass-attention-foundations.md) | `860d4976162dbee47edd0153d0eb8ff8089471d6ff8b7271653e6d8effad7b66` | `ab360def97ca95a553eabcaf66d746e77f05125663b141dbb29a5eeab1aea25a` |

The workshop system record names its base checkout commit; some captured files were uncommitted. Per-file hashes identify the tested working tree. The archived workshop summary remains byte-identical.

## Host-only regression execution

The corrected attention reference compiled with `-Wall -Wextra -Werror` and `ATTENTION_HOST_ONLY`, then ran successfully on arm64. This does not compile its CUDA branch or test a GPU. [Commands, UTC dates and system information](review-validation/2026-11-21-host/results.json) · [Captured output](review-validation/2026-11-21-host/run.log).

## Critical finding dispositions

| Review ID | Correction |
| --- | --- |
| B2-Ch02-1 | Both DSMEM programs rendezvous before remote storage owners exit. GPU rerun pending. |
| B2-Ch02-2 | Multicast sketch distinguishes cluster initialization, per-destination transaction barriers, mask and completion waits. |
| B2-Ch03-1 | Named barriers count all 192 participating producer/consumer threads and separate ready/release generations. |
| B2-Ch04-1 | 128-byte-swizzled BF16 example uses a 64-element contiguous extent; descriptor, byte budget and walkthrough agree. |
| B3-Ch07-1 | BF16-input Hopper WGMMA requires FP32 accumulation; FP16 conversion belongs in the epilogue. |
| B3-Ch07-2 | FP8 pair conversion uses the two-argument intrinsic, raw-half pair and storage return type. CUDA compilation still pending. |
| B3-Ch10-1 | CUTLASS fragments are traversed backwards, distinct from the paper's tile-start-owner protocol. The small role demo does not test that scheduler. |
| B4-Ch11-1 | Optional NVLS is distinguished from ring/tree and does not make eight-endpoint AllReduce a guaranteed one-peer transfer. |
| B4-Ch11-2 | Per-endpoint traffic is separated from aggregate fabric traffic; the false O(N)-to-O(1) claim is removed. |
| B4-Ch11-3 | Ring bandwidth and tree latency characteristics are corrected; selection is per collective. |
| B4-Ch11-17 | Top-K activates K experts; equal-sized top-2 experts cost approximately twice one dense FFN, independent of expert count. |
| B4-Ch11-18 | Tuning starts with NCCL's defaults and observed selection; invented SHARP settings and reversed protocol advice are removed. |
| B4-Ch12-1 | Pending PV completes before O rescaling or V-slot release; the final PV contribution is drained. This is a protocol sketch, not a newly executed FA3 implementation. |

## Other repair areas

- Architecture/ISA: occupancy, bandwidth allocation, FP64/INT32 explanations,
  TMA swizzle/subsampling/OOB/alignment, reduction operation/type combinations,
  WGMMA descriptors and operand precision, FP8 rounding/scaling/loading, and
  asynchronous buffer ownership.
- Scheduling/distributed examples: wave-loss arithmetic, Stream-K grouping,
  spatial tail coverage, ping-pong scope, PDL/prefetch distinctions, GPipe
  dependency timelines, network direction conventions and NCCL/operator scope.
- Teaching chapters: chapter-number mappings, composition-domain distinction,
  explicit limitations of the Python arithmetic models, cooperative/ping-pong
  timelines, Hopper FP8 scaling and accumulation policies, correctness-build
  serialization warnings and framework integration scope.
- Evidence/presentation: historical source labels, base-commit versus working-tree
  identity, B200 CPU/GPU counts and build targets, importer path, Nsight command
  syntax, TOC targets, worked SVG corrections and diagram viewer navigation.

## Deliberate limits and retained records

The redirect stubs are retained to preserve published external URLs; deleting
these is optional housekeeping, not a correctness fix. Archived source copies,
commands, logs, hashes and frozen execution summaries are not rewritten. The
build adds a historical-record notice to those summary pages. The attention
acceptance tolerance remains an explicit acceptance contract, not a claim about
the observed maximum error. Source/log consistency cannot prove hardware
attestation. No missing capture driver, tarball, optimized benchmark, two-SM
UMMA run, PyTorch extension or GPU result has been invented.

Framework integration is documented as a contract, not a compiled extension.
The Python workbooks remain limited teaching models and are no longer cited as
concurrent-protocol tests. Those are scope corrections rather than new execution
results. The FlashAttention implementation-reading reference now names an exact
upstream commit.
