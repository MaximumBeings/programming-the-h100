#!/usr/bin/env python3
"""Align linked figure alt text and existing italic captions with original art.

Only image markup and immediately adjacent single-emphasis figure captions are
edited. Chapter headings, code listings and other explanations stay untouched.
"""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[2]
items=json.loads((ROOT/'tools/illustrations/manifest.json').read_text())
manifest={x['file'].removeprefix('docs/'):x for x in items}
pattern=re.compile(r'\[!\[([^\]]*)\]\((images/ch\d\d/[^)]+\.svg)\)\]\(\2\)')
count=0
for path in (ROOT/'docs').glob('*.md'):
    source=path.read_text();changes=[]
    for m in pattern.finditer(source):
        if m[2] not in manifest:continue
        item=manifest[m[2]]
        replacement=f'[![{item["title"]}]({m[2]})]({m[2]})'
        end=m.end()
        caption=re.match(r'\n\n\*(?!\*)(.+?)\*(?=\n\n|\n?$)',source[end:],re.S)
        if caption:
            old=caption[1]
            prefix=re.match(r'(Figure\s+[\d.]+\s*[—–:.-]\s*)',old)
            links=re.findall(r'\[[^\]]+\]\([^)]+\)',old)
            new=(prefix[0] if prefix else '')+item['lede']
            if links:new+=' '+ ' '.join(dict.fromkeys(links))
            replacement+='\n\n*'+new+'*'
            end+=caption.end()
        changes.append((m.start(),end,replacement));count+=1
    for start,end,replacement in reversed(changes):source=source[:start]+replacement+source[end:]
    if changes:path.write_text(source)
print(f'Aligned {count} linked technical figures with their new art.')
