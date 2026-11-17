#!/usr/bin/env python3
"""Render every SVG and assemble contact sheets for visual inspection."""
from pathlib import Path
import cairosvg
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
output = root / 'validation' / 'diagram-previews'
output.mkdir(parents=True, exist_ok=True)
files = sorted(path for path in (root / 'docs' / 'images').rglob('*.svg') if not path.name.startswith('.'))
for index, source in enumerate(files):
    target = output / (source.parent.name + '-' + source.stem + '.png')
    cairosvg.svg2png(url=str(source), write_to=str(target), output_width=1200)
    if index % 12 == 0:
        sheet = Image.new('RGB', (1600, 1000), '#e2e8f0')
    thumb = Image.open(target).convert('RGB')
    thumb.thumbnail((390, 290))
    col, row = index % 4, (index % 12) // 4
    sheet.paste(thumb, (col * 400 + (400 - thumb.width)//2, row * 333 + 20))
    ImageDraw.Draw(sheet).text((col*400+8, row*333+312), str(source.relative_to(root / 'docs')), fill='#172033')
    if index % 12 == 11 or index == len(files)-1:
        sheet.save(output / f'contact-{index//12+1:02}.png')
print(f'Rendered {len(files)} SVGs.')
