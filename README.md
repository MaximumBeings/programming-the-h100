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

The chapter files are the original markdown, copied verbatim — nothing in them
was edited.

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
