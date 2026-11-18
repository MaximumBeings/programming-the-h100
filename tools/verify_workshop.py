#!/usr/bin/env python3
"""Verify the exact workshop sources, actual captures and rendered chapter records."""
import json
from pathlib import Path
import re
from publish_workshop import ROOT, DATASET, PAGES, sha, validate, section, supplemental

def verify(root=ROOT):
    public=root/'docs/validation'/DATASET
    inventory=json.loads((public/'integrity.json').read_text())
    actual={p.relative_to(public).as_posix() for p in public.rglob('*') if p.is_file()}-{'integrity.json'}
    if inventory.get('dataset')!=DATASET or actual!=set(inventory['files']):raise ValueError('Workshop inventory differs')
    for name,digest in inventory['files'].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or sha(public/rel)!=digest:raise ValueError('Changed/unsafe workshop evidence: '+name)
    summary=root/'docs/cutlass-workshop-validation.md'
    if sha(summary)!=inventory['summary_sha256']:raise ValueError('Workshop summary hash differs')
    rows,system,profiles=validate(public,root)
    extra=supplemental(public,root)
    if extra and extra not in summary.read_text():raise ValueError('Supplemental claims differ from evidence')
    for row in rows:
        stem=Path(row['source']).stem;text=(root/'docs'/PAGES[stem]).read_text()
        parts=re.findall(r'<!-- workshop:start:'+re.escape(stem)+r' -->(.*?)<!-- workshop:end:'+re.escape(stem)+r' -->',text,re.S)
        if len(parts)!=1 or ('<!-- workshop:start:'+stem+' -->'+parts[0]+'<!-- workshop:end:'+stem+' -->')!=section(row,public,root):raise ValueError('Workshop chapter source/commands/dates/output differ: '+stem)
    return rows
if __name__=='__main__':
    rows=verify()
    print('PASS: eight CUDA translation units, two Python references, eleven successful invocations, two real Systems captures, exact sources/dates/output and immutable inventory.')
