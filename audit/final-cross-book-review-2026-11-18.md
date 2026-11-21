# Final cross-book adversarial review — November 18, 2026

> Historical audit, scoped to its stated date and source revision. Later source corrections and outstanding GPU reruns are tracked in [the November 21 review repair record](../docs/review-fixes-2026-11-21.md). This note does not upgrade an old execution to a result for corrected source.

## H100 findings corrected

- Cluster placement is guaranteed within one GPC, not a portable physical-neighbor SM map; cluster size alone does not determine occupancy. DSMEM access requires initialized storage and sufficient target-block lifetime.
- Register operands and local-memory spills are not universal one-cycle/register-versus-HBM guarantees. Cache behavior, instruction dependencies and resource limits matter.
- Host blocking, device completion and asynchronous initiation are distinct. Release/acquire and proxy-fence explanations now state synchronization/scope requirements.
- TMA capabilities are scoped to supported descriptor/instruction combinations; pitch, alignment, layout and bounds still apply.
- Ping-pong scheduling does not guarantee complete overlap or elimination of idle time. Fusion depends on operation dependencies and synchronization scope.
- Four SVG assets had labels using missing embedded-font glyphs: seven ellipsis labels and one opener's alpha/beta notation now use covered ASCII spellings. Generator sources were corrected too.
- The SVG checker now checks embedded glyph coverage across all 175 assets, plus embedded-font application/license, 760-unit bounds and label overlap for the 16 numerical worked figures. Its required Python libraries are declared for clean CI installs.

## Method and limits

Three agents independently reviewed H100 content, multi-GPU content and visual/worked material, followed by parent integration. Content coverage includes the chapter sequence, setup, worked examples, summaries and exercises. Visual checks combine automated inventory/font checks with numerical worked-figure checks, contact sheets and targeted full-size renders; this is not a claim that every legacy figure was manually inspected at full size.

Tested executable sources, downloadable source revisions, raw logs, captured commands and execution records are unchanged. Incorrect non-executable explanatory text diagrams were corrected rather than retained as contradictory teaching material. No GPU host was accessed and no new GPU measurement is implied. Existing source/evidence consistency checks remain authoritative for the precise tested revisions and scope. No new GPU numerical defect was identified during this review.

Installation shell blocks were parsed with Bash without execution. NVIDIA driver, CUDA 12.8 Update 1, NCCL and Fabric Manager documentation were rechecked; the setup guides now distinguish the selected corresponding-driver baseline from the more limited CUDA minor-version compatibility path. Installation on a newly provisioned host was not repeated.

Validation includes the repository's execution-evidence checks, strict MkDocs build, built internal links/anchors, displayed-source preservation and SVG checks. Source commits and Pages deployments are checked after pushing; canonical published articles and SVG/CSS hashes are compared with the final local build.

## Primary references

- [CUDA programming guide, clusters and memory model](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html).
- [CUDA runtime synchronization behavior](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html).
- [CUDA 12.8 Update 1 driver compatibility tables](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-driver).
- [NVIDIA Ubuntu driver installation](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/ubuntu.html).
- [cuBLAS-Xt API](https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublasxt-api).
- [GPipe](https://arxiv.org/abs/1811.06965).

## Local verification result

Local results: 52 HTML pages; 34,082 internal targets/fragments; 97 displayed source listings with preserved tokens/comments; all evidence checks passed. The 18 changed explanatory fences and their before/after hashes are recorded in `final-explanatory-fence-changes-2026-11-18.json`.
