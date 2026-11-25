#!/usr/bin/env python3
"""Verify fresh CPU audit captures without replacing historical GPU evidence."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/validation/2026-11-25-inference-review'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify():
    inventory=json.loads((DATA/'checksums.json').read_text())['files']
    actual={str(p.relative_to(DATA)) for p in DATA.rglob('*') if p.is_file() and p.name!='checksums.json'}
    assert actual==set(inventory),'CPU review inventory changed'
    for name,value in inventory.items():assert digest(DATA/name)==value,f'CPU review capture changed: {name}'
    system=json.loads((DATA/'system.json').read_text())
    assert system['scope']=='CPU references and simulations only; no new GPU execution'
    for name,value in system['source_hashes'].items():
        assert digest(DATA/'sources'/name)==value
        assert digest(ROOT/'docs/examples/inference-systems'/name)==value,f'current source requires rerun: {name}'
    rows=json.loads((DATA/'results.json').read_text())
    assert len(rows)==34,'expected 30 reference invocations plus four regression suites'
    for row in rows:
        assert row['exit_code']==0,f'failed CPU audit command: {row["source"]}'
        assert (DATA/row['log']).stat().st_size>0
        original=ROOT/row['source']
        assert digest(original)==row['source_sha256'],f'current tested source drift: {row["source"]}'
        archived=DATA/('tests' if row['mode']=='regression' else 'sources')/original.name
        assert digest(archived)==row['source_sha256']
    executed={Path(r['source']).name for r in rows if r['mode']=='full'}
    assert len(executed)==16 and not any(name.startswith('ch47') for name in executed)
    assert len([r for r in rows if r['mode']=='regression'])==4
    return executed


if __name__=='__main__':
    tested=verify()
    print(f'PASS: {len(tested)} CPU programs, 34 successful invocations, immutable logs and exact current-source identity; GPU evidence remains historical.')
