#!/usr/bin/env python3
"""Verify the immutable November 18 rerun; never execute CUDA."""
import argparse, hashlib, json, re
from pathlib import Path
from validate_examples import ROOT, programs
NAME = '2026-11-18-two-h100'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(root=ROOT):
    public = root/'docs/validation'/NAME
    inventory = json.loads((public/'integrity.json').read_text())
    actual = {p.relative_to(public).as_posix() for p in public.rglob('*') if p.is_file()}-{'integrity.json'}
    if set(inventory['files']) != actual: raise ValueError('Rerun evidence inventory differs')
    for name, digest in inventory['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or sha(public/rel) != digest:
            raise ValueError('Changed rerun evidence: '+name)
    if sha(root/'docs/revalidation-2026-11-18.md') != inventory['summary_sha256']:
        raise ValueError('Rerun summary hash differs')
    rows=json.loads((public/'results.json').read_text())
    current={(x['page'],x['example']):x['sha256'] for x in programs()}
    recorded={(x['page'],x['example']):x['sha256'] for x in rows}
    if len(rows)!=27 or len(recorded)!=27 or recorded!=current:
        raise ValueError('Rerun must cover all 27 exact current sources')
    passed=blocked=0
    for row in rows:
        folder=public/Path(row['page']).stem/Path(row['example']).stem
        if sha(folder/row['example'])!=row['sha256'] or row['compile_exit']!=0:
            raise ValueError('Rerun source/compilation mismatch: '+row['example'])
        for field in ('timestamp','compile_timestamp','finished_timestamp'):
            if not re.match(r'^2026-11-18T.*\+00:00$',row[field]):raise ValueError('Missing actual UTC stage time')
        output=(folder/'run.log').read_text()
        if row['status']=='PASS':
            if row['run_exit']!=0 or re.search(r'\b(?:FAIL|FAILED)\b',output):
                raise ValueError('Rerun PASS contradicts log: '+row['example'])
            text=(root/row['page']).read_text()
            marker=re.search(r'<!-- revalidation:start:'+re.escape(row['example'])+r' -->(.*?)<!-- revalidation:end:'+re.escape(row['example'])+r' -->',text,re.S)
            if not marker or not all(s in marker[1] for s in (row['sha256'],row['run_timestamp'],row['compile_timestamp'])):
                raise ValueError('Missing dated chapter rerun: '+row['example'])
            shown=re.findall(r'```text\n(.*?)\n```',marker[1],re.S)
            if shown!=['\n'.join(line.rstrip() for line in output.splitlines()).rstrip()]:raise ValueError('Chapter output differs from raw rerun')
            passed+=1
        elif (row['status']=='BLOCKED_HARDWARE' and row['example']=='ex13_hybrid_parallel_training.cu'
              and row['required_gpus']==8 and row['run_exit']=='unexecuted' and 'Requires 8 visible GPU(s); discovered 2.' in output):
            blocked+=1
        else:raise ValueError('Unexpected rerun status')
    if (passed,blocked)!=(26,1):raise ValueError('Expected 26 H100 passes and one eight-GPU prerequisite')
    system=(public/'system-info.log').read_text()
    if system.count('NVIDIA H100 80GB HBM3')<2 or 'V12.8.93' not in system or 'NV18' not in system:
        raise ValueError('Rerun hardware/tool metadata differs')
    attention=(public/'attention/run.log').read_text()
    if len(re.findall(r'^N=\d+ d=\d+ causal=(?:True|False): PASS ',attention,re.M))!=12:
        raise ValueError('Attention rerun lacks 12 successful cases')
    if sha(public/'attention/validate_attention.py')!=sha(root/'tools/validate_attention.py'):
        raise ValueError('Attention source drift')
    variant=json.loads((public/'nonuniform/manifest.json').read_text())
    if variant['status']!='PASSED' or sha(public/'nonuniform'/variant['variant'])!=variant['variant_sha256']:
        raise ValueError('Nonuniform rerun evidence differs')
    for stage in ('preflight','compile','run'):
        evidence=variant[stage]
        if evidence['returncode']!=0 or sha(public/'nonuniform'/evidence['log'])!=evidence['log_sha256']:
            raise ValueError('Nonuniform stage failed or changed')
    for manifest in (public/'correction-memcheck').glob('*/*/memcheck.json'):
        meta=json.loads(manifest.read_text());log=manifest.with_name('memcheck.log')
        if meta['exit_code']!=0 or meta['zero_error_summaries']!=1 or sha(log)!=meta['log_sha256']:
            raise ValueError('Correction memcheck failed or changed')
    if len(list((public/'correction-memcheck').glob('*/*/memcheck.json')))!=4:
        raise ValueError('Expected all four correction memchecks')
    return {(r['page'],r['example']):r for r in rows}

if __name__=='__main__':
    verify(); print('PASS: 27 exact current sources, 26 H100 executions, eight-GPU prerequisite, dated chapter outputs, 12 attention cases and four memchecks.')
