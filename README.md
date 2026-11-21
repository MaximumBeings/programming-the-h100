# Programming the H100: CUDA, CuTe, and CUTLASS from First Principles

A [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) book built
from the H100 companion guides.

```
docs/
  index.md                       landing page + table of contents
  01-h100-architecture.md        Part I    — The Machine
  02-clusters-inline-ptx.md      Part II   — The Instruction Layer
  03-asynchronicity-mbarrier.md
  04-cutensormap-tma-swizzling.md  Part III — Moving Data
  05-cp-async-bulk-l2-hinting.md
  06-wgmma-part-1.md             Part IV   — Tensor Cores
  07-wgmma-part-2.md
  08-kernel-design-part-1.md     Part V    — Kernel Design
  09-kernel-design-part-2.md
  10-stream-k-scheduling.md
  11-multi-gpu-nccl.md           Part VI   — Scaling Out
  12-flashattention-3.md        Part VII  — Putting It to Work
  13-nsight-compute.md
  14-nsight-systems.md
  15-cute-layouts.md                 Introductory → intermediate → advanced
  16-cute-strides.md
  17-cute-layout-algebra.md
  18-cute-tensors-tiling.md
  19-cta-indexing.md
  20-tiled-copy.md
  21-mma-atoms.md
  22-shared-layouts.md
  23-gemm-construction.md
  24-cutlass-gemm.md
  25-cutlass-pipelines.md
  26-epilogue-fusion.md
  27-precision-block-scaling.md
  28-benchmarking-integration.md
  29-workbook-capstone.md
  30-cutlass-attention-foundations.md
  31-cutlass-hopper-fmha.md
  appendix-blackwell-b200.md
  review-fixes-2026-11-21.md
  examples/advanced/            new exercise sources and validation plan
  stylesheets/extra.css          typography + code-block tuning
hooks/fix_anchors.py             build-time link repair (see below)
mkdocs.yml                       site config and nav
```

Historical evidence covers 26 H100 program revisions and a later revised training program on eight A100 GPUs across the recorded H100 and A100 hosts on November 17–18, 2026. The [complete validation summary](docs/final-audit.md) combines 26 successful H100 workloads with the [eight-A100 training execution](docs/eight-gpu-validation.md), which checks every final weight and per-step loss against an independent CPU reference.

The four barrier-ordering corrections in Chapters 3–5 passed the November 18 two-H100 rerun and memory checks. **Sixteen original-program revisions retain matching execution records; eleven corrected revisions need GPU revalidation** across the H100 and eight-A100 hosts. The [additional H100 record](docs/revalidation-2026-11-18.md) includes fresh compilation dates, execution times, system information and sample outputs.

The records retain exact sources, commands, UTC timestamps, topology and raw logs. Original raw archives remain downloadable for provenance. The portable training program targets `sm_80` on A100; Hopper-specific WGMMA and TMA have H100 records. Chapter 12’s executable recurrence reference remains separate from its FA3 skeleton; Chapter 13 distinguishes captured profiles from illustrative diagnoses.

Collect fresh evidence with `python3 tools/validate_examples.py`; each run gets a unique UTC folder under ignored `validation/`. Positional chapter/example filters remain supported. `--list` lists source hashes without execution, `--out PATH` selects a fresh directory and rejects existing contents, and `--compile-only` explicitly labels workloads unexecuted. Compile/runtime failures return 1; hardware prerequisites return 2, including programs that print a prerequisite diagnostic but exit zero.

`tools/publish_final_audit.py` prepares the separate final audit from collected records and refuses to overwrite an existing publication. It checks all 27 source hashes and includes the nonuniform Chapter 7 audit, attention reference and optional captured memcheck log. `--refresh-summary` updates the current combined H100/A100 presentation from verified records while preserving every raw execution row and log. `tools/verify_final_audit.py` verifies the published inventory, source and log hashes, statuses and summary without executing CUDA. The historical publisher `tools/publish_validation.py` is retained for its original workflow; do not use it to replace the preserved dated archive.

`tools/publish_eight_gpu_audit.py` and `tools/verify_eight_gpu_audit.py` publish and verify the separate A100 training supplement. The historical verifiers allow its changed source only when that exact program has a fully verified supplemental PASS record.

`python3 tools/validate_attention.py` runs the online-softmax recurrence checks on CUDA. `tools/audit_nonuniform_epilogue.py --run` builds and runs the stronger Chapter 7 host-input/reference variant while preserving its device implementation. `tools/render_diagrams.py` renders SVGs for inspection (requires CairoSVG and Pillow).

> **November 21 corrections:** historical executions apply to their archived source hashes. Some current examples have subsequent fixes and require fresh GPU validation; see the [repair record](docs/review-fixes-2026-11-21.md). An evidence-verifier pass is an offline consistency check, not a GPU rerun.

## Local preview

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
mkdocs serve
```

Open <http://127.0.0.1:8000>.

## Publishing to GitHub Pages

1. Edit `site_url`, `repo_url` and `repo_name` at the top of `mkdocs.yml` to
   point at your repository.
2. Push to `main`. `.github/workflows/ci.yml` runs `mkdocs gh-deploy --force`,
   which builds the site and pushes it to the `gh-pages` branch.
3. In the repository settings, set **Pages → Source** to the `gh-pages` branch
   (GitHub usually does this automatically on the first deploy).

## About `hooks/fix_anchors.py`

Each chapter opens with a hand-written "Table of Contents" whose links were
written for GitHub's anchor rules — `#5--connectx-7` — which don't match the
anchors python-markdown generates (`#5-connectx-7-and-the-inter-node-bottleneck`).
Rather than editing the source markdown, this hook rewrites those links at build
time by matching each anchor against the page's real heading slugs. It also:

- resolves the abbreviated anchors (`#8--infiniband-topology` →
  the full heading slug);
- rewrites Chapter 8's links to sections §9–§20 as cross-page links into
  Chapter 9, since that lesson is split across two files;
- gives Chapter 9 an `<h1>` from the nav title, since its source file starts
  mid-sequence at `## 9`.

`mkdocs build --strict` checks construction; `python3 tools/verify_site_links.py` checks internal links in the built site.

Built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## Reader tools

Start with `docs/getting-started.md`. `python3 tools/h100_preflight.py` checks CUDA-visible Hopper devices and the installed compiler without PyTorch; `--minimum-gpus 8` checks the prerequisite for the eight-GPU teaching workload. The recorded preflight ran on the H100 PCIe host on November 17, 2026.

`python3 tools/verify_validation_records.py` checks historical source downloads, commands, dates and displayed output against their original records. A current chapter source may differ when a fully verified fresh audit contains its exact hash with a successful compile and workload, or when an exact source revision is registered in `tools/gpu_revalidation.json` or the post-review `tools/review_source_revisions.json` and visibly labeled as awaiting a GPU rerun. Verified corrections require successful records of their exact source hashes; any future pending correction cannot inherit historical PASS status. This applies to both corrected Chapter 11 programs. CI runs both historical and fresh verifiers before building the documentation; these checks do not execute CUDA.

Every chapter includes learning goals, two self-checks with expandable worked answers, and an original SVG opening illustration. `tools/create_chapter_openers.py` regenerates those illustrations.

## Original technical illustrations

All 112 technical SVG figures are generated from original layouts in
`tools/illustrations/catalog.py`, `drawing.py` and `specialized.py`. The 15
chapter-opening illustrations keep their separate design and generator.

```bash
python3 -m pip install -r tools/illustrations/requirements.txt
python3 tools/illustrations/embed_font.py
python3 tools/illustrations/catalog.py
python3 tools/illustrations/sync_captions.py
python3 tools/verify_svg_readability.py
mkdocs build --strict
```

The drawings use a 760-unit reading-column canvas, selectable text and an
embedded Inter font subset with its SIL Open Font License. Architecture paths,
barrier handoffs and network topologies have explicit connections; matrix
layouts and timelines use their own visual forms. Source notes distinguish
product specifications, illustrative models and published paper results.
Regenerating figures does not compile or execute any CUDA example.

## Code listing presentation

Website builds format C, C++ and CUDA listings with pinned clang-format 19.1.7: four-space indentation, 68-column layout and expanded control-flow blocks. The build rejects changes to lexical tokens, literals or comment contents. Only displayed listings are formatted; stored sources, execution hashes, commands and captured outputs remain unchanged. Long literals wrap visually without inserting copied newlines. Style settings are in `.clang-format`; the build hook is `hooks/code_formatting.py`.

## Systems and CUTLASS extension

Chapters 15–29 are sequenced as introductory (15–19), intermediate (20–24) and advanced (25–29). They adapt the supplied tutorial’s useful progression while correcting layout formulas, instruction mappings, framework attribution and precision/performance claims. Chapter 14 adds Nsight Systems.

The [November 18 H100 workshop record](docs/cutlass-workshop-validation.md) captures all ten current workshop sources: eight CUDA translation units, two Python references and eleven passing invocations, with actual sources, commands, system metadata, UTC dates, outputs and two imported Systems reports. The original 27-program evidence remains intact. Use `tools/run_advanced_validation.py` for a new capture or omit execution flags for a side-effect-free plan.

Linked diagrams open in an accessible overlay on the same page. Use Close, Escape or the backdrop to return to the chapter. The viewer also works after instant navigation between chapters.

[Native B200 appendix](docs/appendix-blackwell-b200.md) · [B200 compatibility scope](docs/validation-b200.md)
