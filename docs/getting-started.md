# Getting Started & Installation

[![GETTING STARTED: From an SSH shell to a GPU result](images/chapter-openers/00-getting-started.svg)](images/chapter-openers/00-getting-started.svg)


This guide takes you from a terminal to the book's first verified GPU result. You can read and preview the website on a machine without an NVIDIA GPU. Run the Hopper-specific programs on a compatible NVIDIA GPU host, locally or over SSH.

!!! success "What was actually tested"

    The recorded revisions of all **27 complete CUDA programs compiled and executed successfully across the recorded hosts**. On November 17, 2026, the original validation used **one H100 PCIe**: all 27 complete programs compiled and 24 single-GPU workloads passed. The [follow-up audit](final-audit.md) used **two H100 80GB HBM3 GPUs connected through NV18**, compiling all 27 programs and executing 26 workloads, including peer transfer and NCCL AllReduce. Both hosts used **Ubuntu 24.04.5 LTS**, driver **580.105.08**, CUDA Toolkit **12.8.93**, GCC **13.3.0**, C++17, `-O2`, and `sm_90a`. On November 18 (UTC), the revised portable training program passed on [eight A100 GPUs](eight-gpu-validation.md), targeting `sm_80`, with full final-weight and per-step loss comparisons against a CPU reference. This was not an eight-H100 run.

    The SSH client instructions describe standard usage; the linked system records identify the exact execution configurations. [System record](validation/2026-11-17/system-information.txt) · [Execution manifest](validation/2026-11-17/results.json) · [Preflight JSON](validation/2026-11-17/preflight.json).

The final review added required async-proxy publication fences to four examples in Chapters 3–5. **23 current revisions retain verified PASS records; four corrected revisions await GPU revalidation.** Earlier dated logs refer to their exact archived sources.

Select a diagram to open its full-size SVG.

## 1 — Choose the right machine

| What you have | What you can do | Where the GPU commands run |
| --- | --- | --- |
| Any ordinary laptop or desktop with Python | Read the book, build the website, inspect sources and check evidence hashes | No GPU required for these activities |
| Apple Silicon Mac or a computer without a suitable NVIDIA GPU | Use the machine as an SSH client to a Linux GPU host | In the remote Linux shell, not the local terminal |
| One H100 PCIe or H100 SXM | Run the single-GPU examples; H100 PCIe is the tested SKU | On that GPU host |
| Another Hopper GPU, such as H200 | The `sm_90a` architecture target is relevant, but this is a different hardware configuration | Re-run the checks and record your own results |
| Two or more suitable NVIDIA GPUs | Attempt the relevant Chapter 11 P2P/NCCL examples after checking topology | On the multi-GPU host; count alone does not establish peer access |
| Eight suitable GPUs | Attempt the TP=2 × PP=2 × DP=2 teaching workload | Completed separately on eight A100 GPUs with `sm_80`; see [the supplement](eight-gpu-validation.md) |
| A100 or Blackwell | Read the material, but do not assume the Hopper WGMMA programs will run | A100 lacks WGMMA; Blackwell needs a different Tensor Core implementation |

The `a` in `sm_90a` enables architecture-specific features. It does not mean “this GPU and every later GPU.” Use a provider image with a suitable CUDA toolkit already installed when possible. No GPU rental is required just to read or preview the book.

## 2 — Connect from your computer

If you are already logged into the GPU machine, skip to Step 3. Otherwise use the host address and private key supplied for your instance. Keep the key on your own machine; it does not belong in the book repository.

On Linux or macOS:

```bash
chmod 400 /path/to/your-key.pem
ssh -i /path/to/your-key.pem ubuntu@YOUR_INSTANCE_IP
```

Windows users can run the same `ssh -i ...` command in a terminal with OpenSSH installed, using the Windows path to the key. The POSIX `chmod` command above is for Linux/macOS; Windows key-file permissions use Windows access controls. See [Microsoft's OpenSSH client documentation](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh-overview).

On the first connection, compare the server's host-key fingerprint with your provider's trusted information before accepting it. Once connected, the prompt should refer to the **remote Linux host**. All remaining GPU commands in this guide belong there. CPU-only website commands can also run locally.

## 3 — Check the driver and toolkit separately

Run these on the GPU host:

```bash
nvidia-smi
nvcc --version
python3 --version
gcc --version
git --version
```

`nvidia-smi` checks driver/device visibility. Its “CUDA Version” describes driver compatibility; it is not the installed compiler version. `nvcc --version` identifies the CUDA toolkit that will build your source. The validated host reported driver compatibility with CUDA 13.0 while its installed compiler was CUDA 12.8.93; that is not a contradiction.

If `nvcc` exists outside your `PATH`, use the toolkit location configured for your image or environment module. If the machine genuinely has no toolkit, use [NVIDIA's CUDA 12.8 Linux installation guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html) and select the actual distribution, version and architecture. A preconfigured cloud image may already have a working driver; installing another driver unnecessarily can break that setup. The book's example commands assume a Linux CUDA execution host.

## 4 — Download the source and run preflight

```bash
git clone https://github.com/MaximumBeings/programming-the-h100.git
cd programming-the-h100
python3 tools/h100_preflight.py
python3 tools/h100_preflight.py --json > preflight.json
```

The checker uses the CUDA Driver API to count CUDA-visible devices, including the effect of `CUDA_VISIBLE_DEVICES`. It checks Hopper capability, the compiler, and the availability of optional inspection tools. It needs Python's standard library, not PyTorch. It **does not** prove that a kernel is correct or that multiple GPUs have a suitable interconnect.

Selected lines captured on the validation host:

```text
H100 book preflight: PASS
Visible GPUs: 1; nvcc: 12.8.93
  NVIDIA H100 PCIe; CC 9.0; 81079.1 MiB
```

CUDA-visible memory can be lower than the board's nominal capacity or the management tool's physical-memory report because of reserved resources. Use the actual runtime capacity when allocating buffers.

For a particular distributed exercise, require its actual device count:

```bash
python3 tools/h100_preflight.py --minimum-gpus 2
python3 tools/h100_preflight.py --minimum-gpus 8
```

The eight-GPU training program passed on eight A100 GPUs, including comparisons of every final weight and per-step loss against an independent CPU reference. See the [captured execution and system record](eight-gpu-validation.md).

If you downloaded only the website, the [standalone preflight utility](assets/tools/h100_preflight.py) is also available. Save it, inspect it, then run it on your Linux GPU host.

## 5 — Compile and execute your first verified program

From the repository root, copy the exact tested source to a scratch build directory:

```bash
mkdir -p build/first-run
cp docs/validation/2026-11-17/02-clusters-inline-ptx/ex01_cluster_inspect/ex01_cluster_inspect.cu build/first-run/
cd build/first-run
nvcc -std=c++17 -arch=sm_90a -O2 ex01_cluster_inspect.cu -o ex01_cluster_inspect
./ex01_cluster_inspect
echo $?
```

This is the complete cluster-inspection program from [Chapter 2](02-clusters-inline-ptx.md), not a placeholder fragment. In the captured H100 PCIe run it reported:

```text
DSMEM Exchange Tests: 4/4 passed

Cluster Metadata Verification:
  ✓ All cluster metadata correct

ALL TESTS PASSED — Clusters working correctly!
```

The program's exit status was **0**. GPU `printf` ordering and the device name may differ on another host. Compare the checks and final pass status, not incidental line ordering. [Exact tested source](validation/2026-11-17/02-clusters-inline-ptx/ex01_cluster_inspect/ex01_cluster_inspect.cu) · [Full captured output](validation/2026-11-17/02-clusters-inline-ptx/ex01_cluster_inspect/run.log).

**Learning checkpoint:** For a one-dimensional cluster of two blocks, predict the local rank of global block 3 and its partner block. Check the cluster/rank fields in the program output. The prediction is rank 1 with partner global block 2: global numbering and rank-within-cluster are different quantities. This is a reasoning exercise, not an additional recorded GPU run.

Return to the repository root after this example:

```bash
cd ../..
```

For later examples, use the full command beside that example. Some need `-lcuda`, `-lnccl`, `-lcublas` or additional compilation options; the baseline `nvcc` command is not a universal link command.

## 6 — Install optional profiling tools

Check what the image already provides:

```bash
ncu --version
compute-sanitizer --version
cuobjdump --version
```

On the tested Lambda Ubuntu image, its configured package repository supplied Nsight Compute with:

```bash
sudo apt-get install -y nsight-compute
ncu --version
```

That produced Nsight Compute **2025.1.1.0**. The package command is a record of this image, not a guarantee about every Ubuntu repository. On another system use the vendor's supported installation procedure. The first ordinary-user profile reported `ERR_NVGPUCTRPERM`; profiling with elevated permissions succeeded on the instance we administered. [Chapter 13](13-nsight-compute.md) shows both logs and the exact successful invocation. Do not change system-wide counter permissions merely to imitate a screenshot.

## 7 — Build and read the website locally

These steps need Python, not a GPU. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 tools/verify_validation_records.py
mkdocs build --strict
mkdocs serve
```

Open <http://127.0.0.1:8000> on the same machine. Stop the server with `Ctrl+C`. On Windows, virtual-environment activation has a different path; you can instead invoke the environment's Python directly to install packages and run MkDocs.

If the preview server is on a remote host, keep it bound to localhost and forward it from your own computer:

```bash
ssh -L 8000:127.0.0.1:8000 -i /path/to/your-key.pem ubuntu@YOUR_INSTANCE_IP
```

Start `mkdocs serve` in that remote shell, then open the local address above in your own browser. Alternatively, clone and preview on your laptop so the GPU instance is not needed for documentation work.

The evidence checker compares chapter sources, downloadable sources and recorded hashes. It can detect stale execution records after a code edit, but it does not execute CUDA on a CPU-only CI runner. A strict MkDocs build checks site construction and links. Neither replaces the GPU reference checks.

## 8 — Troubleshoot the first failing step

| What happens | What to check next |
| --- | --- |
| `nvidia-smi` is missing or finds no devices | Confirm you are in the remote GPU shell and that the host has a working NVIDIA driver. A documentation-only laptop is not a CUDA execution host. |
| `nvcc: command not found` | Check the selected toolkit, environment module and `PATH`; the driver alone is not a compiler. |
| Unsupported `sm_90a` target | Check the actual compiler version and its supported targets. Do not remove the `a` to silence a WGMMA assembly error. |
| `no kernel image is available` or architecture mismatch | Match the built binary to the executing GPU. A Hopper-specific binary is not a Blackwell implementation. |
| Undefined `cuTensorMap...`, NCCL or cuBLAS symbols | Use that example's complete compile/link command, and install its development library if absent. |
| Preflight says `BLOCKED` | Read the specific prerequisite error. For multi-GPU work, both CUDA-visible count and topology matter. |
| Illegal access, wrong numbers, timeout or a stalled barrier | Start with the dated source, preserve the first error, compare the full output, and use the chapter's ownership/layout explanation. Run memory checks before performance tuning. |
| `ERR_NVGPUCTRPERM` | Read Chapter 13 and the administrator's profiling policy. It is a counter-access problem, not proof that the kernel is incorrect. |
| Your times differ from the captured output | Compare shape, SKU, clocks, timing scope, warmup and profiler settings. A single-CTA correctness demo is not a peak-throughput benchmark. |

## Where to go next

Choose a [learning path](index.md#how-to-read-this). For the first pass, work through [Chapter 1](01-h100-architecture.md), then the complete examples in [Chapter 2](02-clusters-inline-ptx.md). Each chapter now has learning goals and self-checks with worked answers. Keep your own environment report with every result you want to reproduce.
