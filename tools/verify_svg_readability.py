#!/usr/bin/env python3
"""Check XML validity and the agreed minimum SVG font size without a GPU."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]
paths=sorted((root/'docs/images').rglob('*.svg'))
for path in paths:
    ET.parse(path)
    sizes=[float(v) for v in re.findall(r'font-size(?:\s*:\s*|=")([\d.]+)',path.read_text())]
    if not sizes or min(sizes)<11:
        raise SystemExit(f'FAIL: {path.relative_to(root)}: minimum font must be 11 or larger.')
print(f'PASS: {len(paths)} valid SVGs; every declared font size is at least 11.')
