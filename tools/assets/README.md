Inter variable font, obtained from the Google Fonts Inter package:
https://github.com/google/fonts/tree/main/ofl/inter

Copyright 2020 The Inter Project Authors. See OFL.txt for the SIL Open Font
License 1.1. Each distributed SVG includes the license and an embedded WOFF2
subset; readers do not need a font download or an installed copy of Inter.

To regenerate illustrations, run the relevant generator, then run
`python3 tools/polish_diagrams.py`. Its authoring dependencies are fonttools,
brotli and Pillow; legacy label alignment currently requires Arial at the
macOS path documented in the script. The website build needs none of these
font conversion dependencies. Text stays selectable, searchable and scalable.
