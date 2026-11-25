#!/usr/bin/env python3
"""Capture the complete CPU reference suite after adversarial review.

Does not provision hardware or execute the separate CUDA/vLLM probes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'docs/examples/inference-systems'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();out=args.output.resolve()
    if out.exists():raise SystemExit('Use a new output directory; never overwrite a captured run.')
    out.mkdir(parents=True);(out/'sources').mkdir();(out/'logs').mkdir();(out/'tests').mkdir()
    sources={p.name:p.read_bytes() for p in sorted(SRC.glob('*.py'))}
    for name,data in sources.items():(out/'sources'/name).write_bytes(data)
    jobs=[]
    for name,data in sources.items():
        if name.startswith('ch47'):continue
        n=name[:4]
        flags=['--test'] if n in ['ch32','ch33','ch34','ch35'] else ['--selftest'] if n in ['ch36','ch38','ch39','ch40'] else ['tests'] if n in ['ch41','ch42','ch43','ch44','ch45','ch46'] else []
        if flags:jobs.append((SRC/name,'tests',flags))
        jobs.append((SRC/name,'full',[]))
    tests=['test_inference_boundaries.py','test_inference_core_review.py','test_inference_gateway_review.py','test_inference_engine_review.py']
    for name in tests:
        p=ROOT/'tools/tests'/name;(out/'tests'/name).write_bytes(p.read_bytes());jobs.append((p,'regression',[]))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONUNBUFFERED='1')
    versions=subprocess.check_output([sys.executable,'-c','import numpy, torch; print(numpy.__version__,torch.__version__)'],text=True).strip()
    meta=dict(start_utc=datetime.now(timezone.utc).isoformat(),scope='CPU references and simulations only; no new GPU execution',
              python=sys.version,platform=platform.platform(),machine=platform.machine(),numpy_torch=versions,
              source_hashes={name:sha(data) for name,data in sources.items()},workers=args.workers,
              thread_environment={k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']})
    (out/'system.json').write_text(json.dumps(meta,indent=2))
    def run(job):
        p,mode,flags=job;before=p.read_bytes();cmd=[sys.executable,str(p),*flags];start=datetime.now(timezone.utc).isoformat();t=time.monotonic()
        try:
            r=subprocess.run(cmd,cwd=p.parent,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
            code,log=r.returncode,r.stdout
        except subprocess.TimeoutExpired as e:code,log=124,(e.stdout or b'')+b'\nRUNNER TIMEOUT\n'
        if p.read_bytes()!=before:code,log=125,log+b'\nSOURCE CHANGED DURING EXECUTION\n'
        name=f'{p.stem}-{mode}.log';(out/'logs'/name).write_bytes(log)
        return dict(source=str(p.relative_to(ROOT)),source_sha256=sha(before),mode=mode,command=cmd,
                    start_utc=start,end_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-t,exit_code=code,log='logs/'+name)
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            row=future.result();rows.append(row)
            (out/'results.json').write_text(json.dumps(sorted(rows,key=lambda r:(r['source'],r['mode'])),indent=2))
            print(row['source'],row['mode'],row['exit_code'],flush=True)
    assert all((SRC/name).read_bytes()==data for name,data in sources.items()),'a dependency changed during collection'
    (out/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    inventory={str(p.relative_to(out)):sha(p.read_bytes()) for p in out.rglob('*') if p.is_file()}
    (out/'checksums.json').write_text(json.dumps(dict(collected_utc=datetime.now(timezone.utc).isoformat(),files=inventory),indent=2))
    print(f'{len(rows)} commands; {sum(r["exit_code"]!=0 for r in rows)} failures')
    return int(any(r['exit_code']!=0 for r in rows))


if __name__=='__main__':raise SystemExit(main())
