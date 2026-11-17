# Programming the H100

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
  stylesheets/extra.css          typography + code-block tuning
hooks/fix_anchors.py             build-time link repair (see below)
mkdocs.yml                       site config and nav
```

The chapters include an independent H100 PCIe validation pass executed on November 17, 2026. Dated compiler commands, tested sources, execution logs, system information, and coverage limits are published beside the examples. See `docs/validation/2026-11-17/system-information.txt`.

All 27 complete CUDA programs compile with CUDA 12.8.93; 24 single-GPU workloads pass. The three Chapter 11 multi-GPU workloads still require 2 or 8 GPUs. Chapter 12’s teaching skeleton is distinguished from its executable numerical reference. Chapter 13 includes actual Nsight Compute output. Smaller fragments and illustrations are inventoried rather than claimed as compiled translation units.

`python3 tools/validate_examples.py` extracts complete programs and saves dated evidence; optional arguments select chapter/example names. `python3 tools/validate_attention.py` checks the online-softmax recurrence on CUDA. `tools/publish_validation.py` checks source hashes before publishing captured results. `tools/render_diagrams.py` renders all SVGs for visual inspection (requires CairoSVG and Pillow).

## Local preview

```bash
pip install -r requirements.txt
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

`mkdocs build --strict` passes with zero broken links.

Built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## Reader tools

Start with `docs/getting-started.md`. `python3 tools/h100_preflight.py` checks CUDA-visible Hopper devices and the installed compiler without PyTorch; `--minimum-gpus 8` checks the prerequisite for the eight-GPU teaching workload. The recorded preflight ran on the H100 PCIe host on November 17, 2026.

`python3 tools/verify_validation_records.py` checks that all complete chapter programs and downloadable sources still match their recorded execution hashes. CI runs this before the strict documentation build. It is an evidence-consistency check, not a GPU execution test.

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
