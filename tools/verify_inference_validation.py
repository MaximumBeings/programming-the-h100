#!/usr/bin/env python3
"""Check immutable November 25 evidence and the exact current sources it covers.
This verifier reads captured evidence; it does not execute GPU code.
"""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/validation/2026-11-25-inference'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    inventory=json.loads((DATA/'checksums.json').read_text())['files']
    actual={str(p.relative_to(DATA)) for p in DATA.rglob('*') if p.is_file() and p.name!='checksums.json'}
    assert set(inventory)==actual,'evidence file inventory changed'
    for name,expected in inventory.items():assert sha(DATA/name)==expected,f'capture changed: {name}'
    baseline=json.loads((DATA/'references/results.json').read_text())['results']
    assert len(baseline)==30 and len({r['source'] for r in baseline})==16
    for r in baseline:
        assert r['exit_code']==0 and (DATA/'references'/r['log']).stat().st_size>0
        assert sha(DATA/'sources-baseline'/r['source'])==r['sha256']
    from verify_inference_review import verify as verify_cpu_review
    rerun_sources=verify_cpu_review()
    for p in (DATA/'sources-current').glob('*.py'):
        current=ROOT/('tools/tests' if p.name.startswith('test_') else 'docs/examples/inference-systems')/p.name
        if current.read_bytes()!=p.read_bytes():
            assert p.name in rerun_sources,f'changed GPU probe or unexecuted source needs rerun: {p.name}'
            # verify_cpu_review already requires exact current-source bytes and successful
            # focused/full runs. This transition adds new evidence; old logs stay exact.
    for path in ['paged-cuda.exit','probe.exit','corrected/boundaries-final.exit',
                 'corrected/ch33_moe_expert_parallel.exit','corrected/ch43_paged_kv_cache.exit',
                 'corrected/ch36-tests.exit','corrected/ch36-full.exit']:
        assert (DATA/path).read_text().strip()=='0',f'nonzero final exit: {path}'
    assert 'ALL PASS' in (DATA/'corrected/ch36-full.log').read_text()
    assert 'PASS malformed stream' in (DATA/'corrected/boundaries-final.log').read_text()
    cuda=(DATA/'paged-cuda.log').read_text()
    assert sum(line.startswith('PASS torch.') for line in cuda.splitlines())==12
    assert 'PASS empty-visible-key input rejected' in cuda
    probe=json.loads((DATA/'probe.json').read_text())
    assert probe['source_sha256']==sha(DATA/'sources-current/ch47_vllm_probe.py')
    assert probe['repeated_greedy_equal'] and len(probe['records'])==8
    assert all(r['ok'] and r['ntok']==32 and not r['err'] for r in probe['records'])
    assert probe['summary']['n']==8 and probe['summary']['failures']==0
    assert 'No broken requirements found.' in (DATA/'pip-check.log').read_text()
    assert json.loads((DATA/'system.json').read_text())['model_revisions']==['7ae557604adf67be50417f59c2c2f167def9a775']
    print('PASS: immutable CPU reference captures (30 invocations), corrected source reruns, 12 CUDA comparisons and 8 real vLLM streams; CPU source revisions have separate verified audit reruns; historical CUDA/serving captures remain exact. No capacity or model-quality claim.')

if __name__=='__main__':main()
