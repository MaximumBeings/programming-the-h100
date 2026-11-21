# Installation review — November 18, 2026

> Historical audit, scoped to its stated date and source revision. Later source corrections and outstanding GPU reruns are tracked in [the November 21 review repair record](../docs/review-fixes-2026-11-21.md). This note does not upgrade an old execution to a result for corrected source.

Checked NVIDIA's CUDA 12.8 Update 1 Linux installation guide, current Ubuntu driver guide, NCCL installation guide and Ubuntu 22.04/24.04 x86-64 APT package indexes. Confirmed the CUDA 12.8 toolkit/NVCC, cuBLAS development package, versioned inspection tools and NCCL 2.26.2 CUDA 12.8 packages.

The guides now distinguish fresh Ubuntu hosts from managed cloud images, match repositories to the real OS/CPU, install a driver only when needed, reconnect after reboot, select one toolkit layout, avoid runtime driver stubs, and preserve provider stacks. Chapter setup sections link to the authoritative recipe instead of repeating old CUDA 12.4, purge or Conda-activation instructions.

The two-H100 execution host already supplied its driver/toolkit. It validated the cloud path and full current compilations/runs; this review did not replace that working driver or claim a fresh bare-metal boot test. CUDA installation instructions are checked against vendor documentation and package metadata.

Sources: https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html ; https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/ubuntu.html ; https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html

Systems/CUTLASS extension: NVIDIA Ubuntu2204/Ubuntu2404 package indexes confirm `cuda-nsight-systems-12-8` selects `nsight-systems-2024.6.2` (>=2024.6.2.225). Chapters14–17 pin CUTLASS v4.3.6 commit1afc6d355b8cb992398f17beb4425702f1d57ddd and add directnvcc builds; optional upstream exploration uses CMake/Ninja and explicit target selection. New exercises retain separate planned hardware scope.
