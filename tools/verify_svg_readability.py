#!/usr/bin/env python3
"""Check SVG type; deeply verify the bounded numeric worked-figure family."""
from pathlib import Path
import base64
import io
import re
import xml.etree.ElementTree as ET
from fontTools.ttLib import TTFont
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]


def verify_worked(path, source, svg):
    """Worked assets are untransformed: text positions are canvas coordinates."""
    relative = path.relative_to(ROOT)
    width, height = float(svg.attrib['width']), float(svg.attrib['height'])
    if width != 760 or height <= 0:
        raise ValueError(f'{relative}: expected a positive 760-wide canvas')
    encoded = re.search(r'font/woff2;base64,([A-Za-z0-9+/=]+)', source)
    if not encoded or 'font-license' not in source or 'SIL OPEN FONT LICENSE' not in source:
        raise ValueError(f'{relative}: missing embedded font or license')
    if not re.search(r'(?:font-family\s*:\s*|font-family=")[^;}"<>]*BookDiagram', source):
        raise ValueError(f'{relative}: embedded type is not applied')
    cmap = TTFont(io.BytesIO(base64.b64decode(encoded[1]))).getBestCmap()
    boxes = []
    for element in svg.iter():
        tag = element.tag.rsplit('}', 1)[-1]
        if 'transform' in element.attrib:
            raise ValueError(f'{relative}: transformed geometry needs explicit validation support')
        if tag != 'text':
            continue
        text = ''.join(element.itertext())
        missing = {c for c in text if not c.isspace() and ord(c) not in cmap}
        if missing:
            raise ValueError(f'{relative}: missing embedded glyphs {missing!r}')
        size = float(element.attrib.get('font-size', 20))
        font = ImageFont.truetype(str(ROOT/'tools/assets/Inter.ttf'), round(size*100))
        length = font.getlength(text)/100
        x, y = float(element.get('x', 0)), float(element.get('y', 0))
        anchor = element.get('text-anchor', 'start')
        left = x - (length/2 if anchor == 'middle' else length if anchor == 'end' else 0)
        if left < -1 or left+length > width+1 or y-size < -1 or y+size*.3 > height+1:
            raise ValueError(f'{relative}: text outside canvas: {text!r}')
        # A conservative font-metric overlap check, independent of any generator.
        box = (left, y-size*.76, left+length, y+size*.2, text)
        for previous in boxes:
            overlap_x = min(box[2], previous[2])-max(box[0], previous[0])
            overlap_y = min(box[3], previous[3])-max(box[1], previous[1])
            if overlap_x > 3 and overlap_y > 3:
                raise ValueError(f'{relative}: overlapping labels {previous[4]!r} and {text!r}')
        boxes.append(box)


def main():
    paths = sorted((ROOT/'docs/images').rglob('*.svg'))
    worked = 0
    for path in paths:
        source = path.read_text()
        svg = ET.fromstring(source)
        sizes = [float(v) for v in re.findall(r'font-size(?:\s*:\s*|=")([\d.]+)', source)]
        if not sizes or min(sizes) < 11:
            raise ValueError(f'{path.relative_to(ROOT)}: minimum font must be 11 or larger')
        encoded = re.search(r'font/woff2;base64,([A-Za-z0-9+/=]+)', source)
        if encoded:
            cmap = TTFont(io.BytesIO(base64.b64decode(encoded[1]))).getBestCmap()
            for element in svg.iter():
                if element.tag.rsplit('}', 1)[-1] == 'text':
                    missing = {c for c in ''.join(element.itertext()) if not c.isspace() and ord(c) not in cmap}
                    if missing:
                        raise ValueError(f'{path.relative_to(ROOT)}: missing embedded glyphs {missing!r}')
        if path.parent.name == 'worked':
            verify_worked(path, source, svg)
            worked += 1
    print(f'PASS: {len(paths)} valid SVGs, minimum font 11, embedded glyph coverage; {worked} worked figures pass embedded-font, glyph, bounds and label-overlap checks.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, ET.ParseError) as error:
        raise SystemExit(f'FAIL: {error}')
