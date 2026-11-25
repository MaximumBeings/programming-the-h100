#!/usr/bin/env python3
"""Verify a current-source real-vLLM integration capture, not a capacity claim."""
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/validation/2026-11-25-client-rerun'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    inventory=json.loads((DATA/'checksums.json').read_text())['files']
    actual={str(p.relative_to(DATA)) for p in DATA.rglob('*') if p.is_file() and p.name!='checksums.json'}
    assert actual==set(inventory),'client rerun inventory changed'
    for name,expected in inventory.items():assert sha(DATA/name)==expected,f'capture changed: {name}'
    for name in ['ch36_benchmark_teardown.py','ch47_vllm_probe.py']:
        assert (DATA/'sources'/name).read_bytes()==(ROOT/'docs/examples/inference-systems'/name).read_bytes(),f'client source needs real-server rerun: {name}'
    results=json.loads((DATA/'commands.json').read_text())
    for name in ['probe','open_loop','negative']:
        assert results[name]['exit_code']==0
        assert results[name]['start_utc'] and results[name]['end_utc']
    server=results['server']['command']
    revision='7ae557604adf67be50417f59c2c2f167def9a775'
    assert server[server.index('--revision')+1]==revision
    assert server[server.index('--tokenizer-revision')+1]==revision
    probe=json.loads((DATA/'probe.json').read_text())
    assert probe['source_sha256']==sha(DATA/'sources/ch47_vllm_probe.py')
    assert probe['repeated_greedy_equal'] and len(probe['records'])==8
    assert all(r['ok'] and r['ntok']==32 and not r['err'] for r in probe['records'])
    assert probe['summary']['n']==8 and probe['summary']['failures']==0
    assert len(probe['usage'])==2 and all(u['completion_tokens']==32 for u in probe['usage'])
    rows=[json.loads(line) for line in (DATA/'open-loop.jsonl').read_text().splitlines()]
    assert len(rows)==16 and all(r['ok'] and r['ntok']>0 and not r['err'] for r in rows)
    for r in rows:
        assert all(math.isfinite(r[k]) for k in ['intended','sent','first','end'])
        assert r['end']>=r['first']>=r['sent']>=r['intended']
    sys.path.insert(0,str(ROOT/'docs/examples/inference-systems'))
    import ch36_benchmark_teardown as client
    printed,_=json.JSONDecoder().raw_decode((DATA/'open_loop.log').read_text().lstrip())
    recomputed=client.summarize([client.Rec(**r) for r in rows],warmup=2)
    recomputed.pop('ttft_samples')
    assert printed.keys()==recomputed.keys()
    assert all(math.isclose(printed[k],recomputed[k],rel_tol=1e-12,abs_tol=1e-12) for k in printed),'printed real-server summary differs from raw records'
    neg=json.loads((DATA/'negative.json').read_text())
    assert {r['case'] for r in neg}=={'unknown_model','client_deadline'}
    for case in neg:
        assert case['expected_failure'] and case['passed']
        r=case['record'];assert not r['ok'] and r['err'] and r['end']>=r['intended']
    assert next(r['record']['err'] for r in neg if r['case']=='client_deadline')=='timeout'
    system=json.loads((DATA/'system.json').read_text())
    assert system['vllm']=='0.11.2'
    assert system['model_revision']=='7ae557604adf67be50417f59c2c2f167def9a775'
    assert system['cuda_available'] and 'H100' in system['gpu']
    return probe


if __name__=='__main__':
    verify()
    print('PASS: exact current client/probe against real vLLM on H100; repeated completions, 8 serial streams, 16 open-loop requests, expected HTTP error and timeout; raw logs preserved.')
