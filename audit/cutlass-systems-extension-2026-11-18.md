# Systems and CUTLASS extension — November 18, 2026

> Historical audit, scoped to its stated date and source revision. Later source corrections and outstanding GPU reruns are tracked in [the November 21 review repair record](../docs/review-fixes-2026-11-21.md). This note does not upgrade an old execution to a result for corrected source.

Added Chapters 14–17: Nsight Systems application timelines, CuTe C++ layout algebra, a pinned CUTLASS Hopper GEMM, and pipeline/fusion/profiler source reading. Updated navigation, the homepage contents and reading map, installation cross-links and the former Chapter13 ending. Added four original SVG chapter maps.

Reference baseline: NVIDIA CUTLASS v4.3.6, commit `1afc6d355b8cb992398f17beb4425702f1d57ddd`; CUDA12.8; C++17; Hopper sm_90a. Ubuntu2204/Ubuntu2404 NVIDIA metadata confirms cuda-nsight-systems-12-8 selects nsight-systems-2024.6.2. Current tool version is recorded independently of NVCC/Compute.

Adversarial review checked actual collective-builder defaults/arguments, element versus byte alignment, B's logical(N,K,L) shape, independent row-major CPU oracle, nonzero beta, batch and edge cases, exact binary-rational arithmetic bounds, NaN output sentinels, stream-slot ownership and event fan-out/fan-in. Corrected collector timeouts to terminate their owned process group and require numerical markers plus an existing profiler report and captured transform kernel summary.

Documentation review checked upstream target names, all seven example49 result labels versus its non-authoritative process exit, compatible schedule pairings, TMA pipeline phase/byte/participant contracts, epilogue carveout, explicit cuBLAS/provider configuration, and nonempty profiler enumeration. Systems coverage distinguishes CPU submission, GPU execution, summaries, actual overlap, critical paths, cross-rank timing limitations and unprofiled baselines.

Offline checks: strict MkDocs build;24builtHTMLpages/internal links;39formattedcode listings with tokens/comments preserved;132SVGXML/minimum-font checks and rendered review of the four new maps;12newBashblocks passsyntaxchecks;Pythoncollector compilation and side-effect-free plan checks. Original27source/record/evidence verification remains passing and unchanged.

Hardware scope: new layout/GEMM/overlap sources have not yet been compiled with CUDA or executed on the next Hopper instance. The separate advanced collector is prepared to capture current source copies/hashes, exact commands, UTC dates, hardware/software metadata, numerical results, and optional Systems reports. Expected checks are labeled as specifications, not fabricated transcripts. No CUTLASS throughput or overlap improvement has been claimed.
