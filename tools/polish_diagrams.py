#!/usr/bin/env python3
"""Embed OFL-licensed Inter and improve SVG label sizing without rasterizing text.

Run with fonttools, brotli and Pillow installed. Re-running is safe. Regenerate
chapter openers first when needed, then run this tool to embed their typography.
"""
from pathlib import Path
from io import BytesIO
from html import unescape, escape
import base64
import re
import xml.etree.ElementTree as ET
from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / 'docs/images').rglob('*.svg'))
MARKER = '<!-- typography: inter-v1 -->'
FONT = ROOT / 'tools/assets/Inter.ttf'

def run():
    # One shared subset ensures every standalone image remains self-contained,
    # including when loaded through <img>, which blocks external font requests.
    chars = ''.join(''.join(t.itertext()) for p in FILES for t in ET.fromstring(p.read_text()).iter('{http://www.w3.org/2000/svg}text'))
    font = TTFont(FONT)
    font = instantiateVariableFont(font, {'opsz': 14}, inplace=True)
    options = subset.Options()
    options.flavor = 'woff2'
    sub = subset.Subsetter(options=options)
    sub.populate(text=chars)
    sub.subset(font)
    font.flavor = 'woff2'
    font.recalcTimestamp = False
    out = BytesIO()
    font.save(out)
    data = base64.b64encode(out.getvalue()).decode()
    license_metadata = '<metadata id="font-license">'+escape('\n'.join(line.rstrip() for line in (ROOT / 'tools/assets/OFL.txt').read_text().splitlines()))+'</metadata>'
    css = ('<style type="text/css">@font-face{font-family:BookDiagram;'
           f'src:url(data:font/woff2;base64,{data}) format("woff2");'
           'font-style:normal;font-weight:100 900} '
           'text{font-family:BookDiagram,Inter,Arial,sans-serif;'
           'font-synthesis:none;text-rendering:optimizeLegibility}</style>')
    original_font = '/System/Library/Fonts/Supplemental/Arial.ttf'
    if not Path(original_font).exists():
        raise SystemExit('Sizing the original Arial labels requires Arial; use a host with this font installed.')
    changed = 0
    for path in FILES:
        src = path.read_text()
        if 'original illustration system: h100-editorial-v1' in src:
            continue
        if MARKER in src:
            src = re.sub(r'<style type="text/css">@font-face.*?</style>', lambda _: css, src, flags=re.S)
            if 'id="font-license"' in src:
                src = re.sub(r'<metadata id="font-license">.*?</metadata>', lambda _: license_metadata, src, flags=re.S)
            else:
                src = src.replace(MARKER, MARKER+'\n'+license_metadata)
            path.write_text(src)
            continue
        # Existing translated labels use a left edge calculated for the old
        # font. Keep their optical center while increasing label size.
        def label(match):
            tag, body = match.group(1), match.group(2)
            size_match = re.search(r'font-size:\s*([\d.]+)px', tag)
            if size_match:
                size = float(size_match[1])
                larger = size * (1.10 if size <= 10 else 1.035)
                tag = tag[:size_match.start(1)] + f'{larger:.3f}' + tag[size_match.end(1):]
                plain = unescape(re.sub('<[^>]+>', '', body))
                translate = re.search(r'translate\(([\d.-]+) ([\d.-]+)\)', tag)
                if translate and 'text-anchor' not in tag and plain:
                    old = ImageFont.truetype(original_font, round(size*100), layout_engine=ImageFont.Layout.BASIC)
                    new = ImageFont.truetype(str(FONT), round(larger*100), layout_engine=ImageFont.Layout.BASIC)
                    shift = (old.getlength(plain)-new.getlength(plain))/200
                    x = float(translate[1])+shift
                    tag = tag[:translate.start(1)]+f'{x:.5f}'+tag[translate.end(1):]
                tag = re.sub(r'font-style:\s*italic;', 'font-style: normal;', tag)
                tag = re.sub(r'font-weight:\s*700;', 'font-weight: 650;', tag)
                if 'font-weight:' not in tag:
                    tag = tag.replace('style="', 'style="font-weight: 450; ', 1)
            # Override old inline families so no system font substitution is
            # necessary. Preserve mathematical tspans and all label wording.
            tag = re.sub(r'font-family:\s*[^;"<>]+', 'font-family: BookDiagram, Inter, Arial, sans-serif', tag)
            tag = re.sub(r'font-family="[^"]+"', 'font-family="BookDiagram, Inter, Arial, sans-serif"', tag)
            return '<text'+tag+'>'+body+'</text>'
        src = re.sub(r'<text([^>]*)>(.*?)</text>', label, src, flags=re.S)
        src = src.replace('#555555', '#475569').replace('#666666', '#475569')
        pos = src.index('>')+1
        if src.startswith('<?xml'):
            pos = re.search(r'<svg\b[^>]*>', src).end()
        src = src[:pos]+'\n'+MARKER+'\n'+license_metadata+'\n'+css+'\n'+src[pos:]
        ET.fromstring(src)
        path.write_text(src)
        changed += 1
    print(f'Polished {changed} SVGs; embedded font subset {len(out.getvalue())} bytes.')

if __name__ == '__main__':
    run()
    from enforce_svg_readability import enforce
    enforce()
