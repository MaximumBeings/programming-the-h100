# Original H100 technical illustrations

`catalog.py` is the content inventory for all 112 technical diagrams. It uses
`drawing.py` for typography, cards, matrices, tables and timelines, then
`specialized.py` for architecture, dataflow, multicast, barrier and topology
relationships. `manifest.json` records the generated figure identifiers and
reading-column dimensions. Existing chapter-opening art is not regenerated.

Generate the embedded font with `embed_font.py`, then run `catalog.py`.
`sync_captions.py` aligns linked figure alt text and existing italic captions
with the new descriptions. The generator asserts unique and complete coverage.

Design conventions:

- White canvas; navy text; restrained teal, blue, violet and amber accents.
- A fixed 760-unit width; height grows with content instead of shrinking labels.
- Titles 29, body 20, data labels typically 17–20, source notes 14.
- Explicit connector paths for physical/logical relationships; numbered
  timeline coordinates; labelled mathematical models.
- No vendor figure is traced or embedded. The Inter font is the only bundled
  third-party visual resource; its license is in every distributed SVG.

Model statements and published benchmark results must not be relabelled as
execution results from the book. CUDA evidence is checked separately by
`tools/verify_validation_records.py`.
