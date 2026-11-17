#!/usr/bin/env python3
"""Enforce a 11-unit SVG label floor and space multiline labels for it.

Requires Pillow for Inter glyph metrics. SVG units are kept at >= 1 CSS pixel
by hooks/diagram_readability.py, including on narrow screens (scroll to pan).
"""
from pathlib import Path
import re
from html import unescape
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
MINIMUM = 11.0
SIZE = re.compile(r'font-size(?:\s*:\s*|=")([\d.]+)')
TRANSLATE = re.compile(r'translate\(([\d.-]+)[ ,]+([\d.-]+)\)')
FONT = ROOT/'tools/assets/Inter.ttf'

def enforce():
    changed = labels = 0
    for p in sorted((ROOT/'docs/images').rglob('*.svg')):
        source = p.read_text()
        if not any(float(s) < MINIMUM for s in SIZE.findall(source)):
            continue
        def group(m):
            block = m[0]
            grew = False
            def label(t):
                nonlocal grew, labels
                tag, body = t[1], t[2]
                match = SIZE.search(tag)
                if not match or float(match[1]) >= MINIMUM:
                    return t[0]
                grew = True
                labels += 1
                old_size = float(match[1])
                tag = tag[:match.start(1)]+f'{MINIMUM:g}'+tag[match.end(1):]
                pos = TRANSLATE.search(tag)
                if pos and 'text-anchor' not in tag:
                    plain = unescape(re.sub('<[^>]+>','',body))
                    old = ImageFont.truetype(str(FONT),round(old_size*100),layout_engine=ImageFont.Layout.BASIC)
                    new = ImageFont.truetype(str(FONT),round(MINIMUM*100),layout_engine=ImageFont.Layout.BASIC)
                    shift = (old.getlength(plain)-new.getlength(plain))/200
                    tag = tag[:pos.start(1)]+f'{float(pos[1])+shift:.5f}'+tag[pos.end(1):]
                    pos = TRANSLATE.search(tag)
                    tag = tag[:pos.start(2)]+f'{float(pos[2])+(MINIMUM-old_size)*.32:.5f}'+tag[pos.end(2):]
                return '<text'+tag+'>'+body+'</text>'
            block = re.sub(r'<text([^>]*?)(?<!/)>(.*?)</text>',label,block,flags=re.S)
            if grew:
                positions=list(TRANSLATE.finditer(block))
                # A Matplotlib text group contains the lines of one label.
                # Keep its vertical center and expand tight baseline spacing.
                if len(positions)>1:
                    ys=[float(t[2]) for t in positions]
                    if all(a<b for a,b in zip(ys,ys[1:])):
                        gaps=[b-a for a,b in zip(ys,ys[1:])]
                        if min(gaps)<12.65:
                            new=[0]
                            for gap in gaps:new.append(new[-1]+max(gap,12.65))
                            shift=(ys[0]+ys[-1]-new[-1])/2
                            for t,y in reversed(list(zip(positions,new))):
                                block=block[:t.start(2)]+f'{y+shift:.5f}'+block[t.end(2):]
            return block
        # Group first, then handle native SVGs whose labels are not grouped.
        source=re.sub(r'<g id="text_\d+">.*?</g>',group,source,flags=re.S)
        source=re.sub(r'<text\b.*?</text>',group,source,flags=re.S)
        # Self-closing empty text lines must not consume the following label.
        # Also enforce the floor directly on every text opening tag.
        def opening(m):
            tag=SIZE.sub(lambda size: size[0].replace(size[1],f'{max(MINIMUM,float(size[1])):g}',1),m[0])
            tag=re.sub(r'font-family:\s*[^;"<>]+','font-family: BookDiagram, Inter, Arial, sans-serif',tag)
            return tag
        source=re.sub(r'<(?:text|tspan)\b[^>]*>',opening,source)
        p.write_text(source)
        changed += 1
    print(f'Enforced minimum {MINIMUM:g}: enlarged {labels} labels in {changed} SVGs.')

if __name__=='__main__':enforce()
