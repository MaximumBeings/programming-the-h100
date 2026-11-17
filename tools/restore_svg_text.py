#!/usr/bin/env python3
"""Restore selectable labels from Matplotlib's preserved text comments."""
from html import escape, unescape
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
for path in (root / 'docs' / 'images').rglob('*.svg'):
    source = path.read_text()
    if '<text' in source:
        continue
    replacements = []
    for start in re.finditer(r'<g id="text_\d+">', source):
        depth = 1
        for tag in re.finditer(r'<g\b[^>]*>|</g>', source[start.end():]):
            depth += -1 if tag[0] == '</g>' else 1
            if depth == 0:
                end = start.end() + tag.end()
                break
        group = source[start.end():end-len('</g>')]
        labels = list(re.finditer(r'<!-- (.*?) -->\s*<g([^>]*)>', group, re.S))
        if not labels:
            raise ValueError(f'No label preserved: {path} {start[0]}')
        output = [start[0]]
        for index, label in enumerate(labels):
            attributes = label[2]
            transform = re.search(r'transform="([^"]+)"', attributes)[1]
            scale = re.search(r'scale\(([\d.]+) -[\d.]+\)', transform)
            if not scale:
                raise ValueError(f'Unexpected transform: {path} {transform}')
            size = float(scale[1]) * 100
            transform = transform[:scale.start()].strip()
            segment = group[label.end():labels[index+1].start() if index+1 < len(labels) else len(group)]
            weight = '700' if 'Sans-Bold' in segment else '400'
            style = 'italic' if 'Oblique' in segment else 'normal'
            color_match = re.search(r'fill: (#[0-9a-fA-F]+)', attributes)
            color = color_match[1] if color_match else '#172033'
            text = unescape(label[1])
            text = re.sub(r'\$_([0-9]+)\$', lambda m: m[1].translate(str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')), text)
            output.append(f'<text transform="{transform}" font-family="Arial, Liberation Sans, sans-serif" font-size="{size:g}" font-weight="{weight}" font-style="{style}" fill="{color}">{escape(text)}</text>')
        output.append('</g>')
        replacements.append((start.start(), end, '\n'.join(output)))
    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    path.write_text(source)
    print(path.relative_to(root), len(replacements), 'selectable label groups')
