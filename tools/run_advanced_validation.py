#!/usr/bin/env python3
"""Capture new Chapters 14–29 exercises separately from the original 27 programs.

No network access or package installation. Without --execute, print the plan only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PIN = '1afc6d355b8cb992398f17beb4425702f1d57ddd'

def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def capture(command, path, timeout=600):
    start = utc(); t = time.monotonic()
    with path.open('w') as log:
        log.write('$ '+shlex.join(map(str, command))+'\n'); log.flush()
        try:
            process = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try: code = process.wait(timeout)
            except subprocess.TimeoutExpired:
                # nvcc and nsys may spawn children. Terminate our complete session,
                # including children that outlive their immediate parent.
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: process.wait(5)
                except subprocess.TimeoutExpired: pass
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
                code = 'timeout'
        except OSError as error:
            log.write(str(error)+'\n'); code = 'launch_error'
    return dict(command=list(map(str, command)), exit_code=code, started_utc=start,
                finished_utc=utc(), seconds=time.monotonic()-t, log=path.name, log_sha256=sha(path))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cutlass',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--profiles',action='store_true',help='Also capture two Nsight Systems traces; requires --execute.')
    args=ap.parse_args()
    cutlass=args.cutlass.resolve();out=args.output.resolve()
    if args.profiles and not args.execute: ap.error('--profiles requires --execute')
    if not args.execute:
        print('PLAN ONLY: compile 8 CUDA translation units, byte-compile 2 Python references; run 11 workloads.')
        print('No files created; no compiler, GPU or profiler executed. Add --execute on the Hopper host.')
        return 0
    revision=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(cutlass),'status','--porcelain','--untracked-files=no'],text=True)
    if revision!=PIN or dirty: raise SystemExit('CUTLASS must be the clean pinned v4.3.6 revision.')
    out.mkdir(parents=True,exist_ok=False)
    system={'utc':utc(),'platform':platform.platform(),'cutlass_sha':revision,'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'commands':[]}
    for i,command in enumerate([['nvcc','--version'],['g++','--version'],['nvidia-smi','--query-gpu=name,uuid,compute_cap,driver_version,memory.total','--format=csv'],['nvidia-smi','topo','-m'],['nsys','--version'],['ncu','--version'],['compute-sanitizer','--version'],['uname','-a'],['lscpu'],['cat','/etc/os-release'],['python3','--version'],['QdstrmImporter','--version']]):
        system['commands'].append(capture(command,out/f'system-{i}.log',45))
    system['source_tree_dirty']=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=normal'],text=True).strip())
    (out/'system.json').write_text(json.dumps(system,indent=2)+'\n')
    gpu_log=(out/'system-2.log').read_text()
    if system['commands'][2]['exit_code']!=0 or not re.search(r',\s*9\.0\s*,',gpu_log):
        raise SystemExit('No recorded Hopper CC 9.0 device; no workloads launched.')
    programs=[
        ('cute_layouts',[((),r'LAYOUT PASS:.*errors=0')]),
        ('cute_algebra',[((),r'ALGEBRA PASS:.*errors=0')]),
        ('cute_fragments',[((),r'FRAGMENT PASS.*errors=0')]),
        ('cute_copy',[((),r'COPY PASS: 3 cases, errors=0')]),
        ('tiled_gemm',[((),r'TILED GEMM PASS: 3 cases, errors=0')]),
        ('cutlass_gemm',[((),r'GEMM PASS: 3 full CPU-reference cases')]),
        ('cutlass_gemm_relu',[((),r'FUSED GEMM PASS: 3 full CPU-reference cases')]),
        ('nsys_overlap',[(('serial',),r'OVERLAP PASS: mode=serial.*errors=0'),(('overlap',),r'OVERLAP PASS: mode=overlap.*errors=0')]),
        ('indexing_workbook',[((),r'WORKBOOK PASS: 7 CPU-only reference groups')]),
        ('fp4_reference',[((),r'FP4 REFERENCE PASS')])]
    records=[]
    for name,runs in programs:
        python_source=name in {'indexing_workbook','fp4_reference'}
        source=ROOT/'docs/examples/advanced'/(name+('.py' if python_source else '.cu'))
        (out/source.name).write_bytes(source.read_bytes())
        binary=out/name
        if python_source:
            command=['python3','-m','py_compile',source]
        else:
            command=['nvcc','-std=c++17','-O2','-lineinfo','-arch=sm_90a','--expt-relaxed-constexpr',
                     '-I'+str(cutlass/'include'),'-I'+str(cutlass/'tools/util/include'),source,'-o',binary]
            if name in {'cutlass_gemm','cutlass_gemm_relu'}:command+=['-lcuda']
        row={'source':str(source.relative_to(ROOT)),'sha256':sha(source),'kind':'python_cpu_reference' if python_source else 'cute_host_oracle' if name in {'cute_layouts','cute_algebra','cute_fragments'} else 'cuda_workload','compile':capture(command,out/(name+'-compile.log')),'runs':[]}
        if row['compile']['exit_code']==0:
            for arguments,marker in runs:
                entry=capture((['python3',source] if python_source else [binary])+list(arguments),out/(name+('-'+arguments[0] if arguments else '')+'-run.log'),120)
                text=(out/entry['log']).read_text()
                extra=(name not in {'cutlass_gemm','cutlass_gemm_relu'} or len(re.findall(r'^(?:FUSED )?GEMM CASE PASS ',text,re.M))==3)
                entry['status']='PASS' if entry['exit_code']==0 and re.search(marker,text) and extra and not re.search(r'\b(?:FAIL|ERROR)\b',text) else 'FAIL'
                row['runs'].append(entry)
        records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
    passed=all(r['compile']['exit_code']==0 and len(r['runs'])==len(runs) and all(x['status']=='PASS' for x in r['runs']) for r,(_,runs) in zip(records,programs))
    profiles=[]
    if passed and args.profiles:
        for mode in ('serial','overlap'):
            prefix=out/('timeline-'+mode)
            command=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                     '--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(prefix),out/'nsys_overlap',mode]
            result=capture(command,out/('nsys-'+mode+'.log'),180)
            report=prefix.with_suffix('.nsys-rep')
            intermediate=prefix.with_suffix('.qdstrm')
            # Some distro packages split the matching host importer from the
            # target CLI. Preserve its original diagnostic and capture the
            # documented manual import as a separate, dated command.
            importer=shutil.which('QdstrmImporter')
            if result['exit_code']==0 and not report.is_file() and intermediate.is_file() and importer:
                result['import']=capture([importer,'--input-file',intermediate,'--output-file',report],out/('nsys-'+mode+'-import.log'),180)
            result['report_exists']=report.is_file()
            text=(out/result['log']).read_text()
            marker=rf'^OVERLAP PASS: mode={mode} slots={1 if mode=="serial" else 2} checked=4194304 errors=0 device_span_ms='
            result['workload_pass']=bool(re.search(marker,text,re.M)) and not bool(re.search(r'\b(?:FAIL|ERROR)\b',text))
            result['captured_kernel']=False
            if result['exit_code']==0 and result['report_exists']:
                stats=capture(['nsys','stats','--report','cuda_gpu_kern_sum','--format','csv',prefix.with_suffix('.nsys-rep')],out/('nsys-'+mode+'-stats.log'),120)
                result['stats']=stats
                result['captured_kernel']=stats['exit_code']==0 and 'transform' in (out/stats['log']).read_text()
            result['status']='PASS' if result['exit_code']==0 and result.get('import',{}).get('exit_code',0)==0 and result['report_exists'] and result['workload_pass'] and result['captured_kernel'] else 'FAIL'
            profiles.append(result)
        (out/'profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
        passed=len(profiles)==2 and all(p['status']=='PASS' for p in profiles)
    inventory={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in {r[0] for r in programs}}
    (out/'integrity.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print('ADVANCED PASS' if passed else 'ADVANCED FAIL',out)
    return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
