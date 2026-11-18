#!/usr/bin/env python3
"""Publish exact successful workshop captures without running or inferring GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil

ROOT=Path(__file__).resolve().parents[1]
DATASET='2026-11-18-h100-workshop'
PAGES={
 'nsys_overlap':'14-nsight-systems.md','cute_layouts':'15-cute-layouts.md',
 'cute_algebra':'17-cute-layout-algebra.md','indexing_workbook':'19-cta-indexing.md',
 'cute_copy':'20-tiled-copy.md','cute_fragments':'21-mma-atoms.md',
 'tiled_gemm':'23-gemm-construction.md','cutlass_gemm':'24-cutlass-gemm.md',
 'cutlass_gemm_relu':'26-epilogue-fusion.md','fp4_reference':'27-precision-block-scaling.md'}
MARKERS={
 'nsys_overlap':'OVERLAP PASS:', 'cute_layouts':'LAYOUT PASS:', 'cute_algebra':'ALGEBRA PASS:',
 'indexing_workbook':'WORKBOOK PASS: 7 CPU-only reference groups', 'cute_copy':'COPY PASS: 3 cases, errors=0',
 'cute_fragments':'FRAGMENT PASS', 'tiled_gemm':'TILED GEMM PASS: 3 cases, errors=0',
 'cutlass_gemm':'GEMM PASS: 3 full CPU-reference cases',
 'cutlass_gemm_relu':'FUSED GEMM PASS: 3 full CPU-reference cases', 'fp4_reference':'FP4 REFERENCE PASS:'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def output(path):
    text=path.read_text();text=text.split('\n',1)[1] if text.startswith('$ ') else text
    return '\n'.join(l.rstrip() for l in text.splitlines()).rstrip()

def command(args,source):
    result=[]
    for index,arg in enumerate(args):
        p=Path(arg)
        if arg.startswith('-I') and arg.endswith('/tools/util/include'):result.append('-I "$CUTLASS_ROOT/tools/util/include"')
        elif arg.startswith('-I') and arg.endswith('/include'):result.append('-I "$CUTLASS_ROOT/include"')
        elif p.name==source:result.append(shlex.quote(source))
        elif p.name==Path(source).stem and index>0:result.append('./example')
        elif index==0 and p.name==Path(source).stem:result.append('./example')
        else:result.append(shlex.quote(arg))
    return ' '.join(result)

def stage_ok(folder,stage,allowed_exit_codes=(0,)):
    path=folder/stage['log']
    if (stage['exit_code'] not in allowed_exit_codes or sha(path)!=stage['log_sha256'] or
        not path.read_text().startswith('$ '+shlex.join(stage['command'])+'\n')):
        raise ValueError('Stage command/status/hash differs: '+path.name)
    for field in ('started_utc','finished_utc'):
        if not re.fullmatch(r'2026-11-18T.*(?:\+00:00|Z)',stage[field]):raise ValueError('Missing actual UTC stage time')

def validate(folder,root=ROOT):
    rows=json.loads((folder/'results.json').read_text())
    if len(rows)!=10 or {Path(r['source']).stem for r in rows}!=set(PAGES):raise ValueError('Expected all ten workshop sources')
    if sum(len(r['runs']) for r in rows)!=11:raise ValueError('Expected eleven workload invocations')
    for row in rows:
        name=Path(row['source']).name;stem=Path(name).stem
        if sha(root/row['source'])!=row['sha256'] or sha(folder/name)!=row['sha256']:raise ValueError('Current/captured source differs: '+name)
        stage_ok(folder,row['compile'])
        if len(row['runs'])!=(2 if stem=='nsys_overlap' else 1):raise ValueError('Wrong run count')
        for run in row['runs']:
            stage_ok(folder,run);text=output(folder/run['log'])
            if run['status']!='PASS' or MARKERS[stem] not in text or re.search(r'\b(?:FAIL|ERROR)\b',text):raise ValueError('Workload not verified: '+name)
    system=json.loads((folder/'system.json').read_text())
    if system['cutlass_sha']!='1afc6d355b8cb992398f17beb4425702f1d57ddd':raise ValueError('CUTLASS pin differs')
    gpu=output(folder/'system-2.log')
    if 'NVIDIA H100 PCIe' not in gpu or not re.search(r',\s*9\.0\s*,',gpu):raise ValueError('H100 system evidence missing')
    profiles=json.loads((folder/'profiles.json').read_text())
    if len(profiles)!=2:raise ValueError('Expected two profile captures')
    for profile in profiles:
        stage_ok(folder,profile)
        if profile['status']!='PASS' or not profile.get('workload_pass') or not profile.get('captured_kernel'):raise ValueError('Profile not validated')
        stage_ok(folder,profile['stats'])
        if 'transform' not in output(folder/profile['stats']['log']):raise ValueError('No captured kernel')
        for key in ('import','importer'):
            if isinstance(profile.get(key),dict):stage_ok(folder,profile[key])
    return rows,system,profiles

def section(row,public,root=ROOT):
    source=Path(row['source']);stem=source.stem;page=PAGES[stem]
    prefix='../' if '/' in page else ''
    linkbase='validation/'+DATASET+'/'
    lines=[f'<!-- workshop:start:{stem} -->','## Complete source and captured execution','',
           f"**Executed on November 18, 2026 (UTC).** Scope: **{ {'cuda_workload':'CUDA workload with its stated numerical checks','cute_host_oracle':'CPU execution of real CuTe coordinate/trait checks','python_cpu_reference':'CPU-only Python arithmetic/format reference'}[row['kind']]}**.",'',
           f"Tested source SHA-256: `{row['sha256']}`. [Exact source]({linkbase}{source.name}) · [System information]({linkbase}system.json) · [Workshop results](cutlass-workshop-validation.md).",'',
           'The listing below matches the tested source. Its presentation is formatted at build time; downloads retain the captured bytes.','',
           '```'+('python' if source.suffix=='.py' else 'cuda')+'\n'+(root/source).read_text().rstrip()+'\n```','',
           f"{'Python byte-compilation' if source.suffix=='.py' else 'CUDA compilation'} UTC: `{row['compile']['started_utc']}`. Exit: `0`. [Compile log]({linkbase}{row['compile']['log']}).",'',
           'Commands use scratch-directory filenames; raw logs retain exact paths. The collector’s CUTLASS include flags are configuration for that capture; the minimal standalone overlap/SIMT commands do not require those headers.','',
           '```bash\n'+command(row['compile']['command'],source.name)+'\n```','']
    for run in row['runs']:
        lines += [f"Executed UTC: `{run['started_utc']}`. Exit: `0`. [Run log]({linkbase}{run['log']}).",'',
                  '```bash\n'+command(run['command'],source.name)+'\n```','', '**Actual captured output**','',
                  '```text\n'+output(public/run['log'])+'\n```','']
    lines += [f'<!-- workshop:end:{stem} -->']
    return '\n'.join(lines)

def supplemental(public,root=ROOT):
    base=public/'supplemental-captures'
    if not base.exists():return ''
    rows={Path(r['source']).stem:r for r in json.loads((public/'results.json').read_text())}
    allowed={r['sha256'] for r in rows.values()};checks=[];analyses=[]
    for manifest in sorted(base.rglob('commands.json')):
        for record in json.loads(manifest.read_text()):
            kind=record['kind'];folder=manifest.parent
            permitted=(0,1) if kind=='nsight_compute' else (0,)
            stage_ok(folder,record,permitted)
            if record.get('source_sha256') and record['source_sha256'] not in allowed:raise ValueError('Supplemental source differs')
            text=output(folder/record['log'])
            if kind=='memcheck':
                stem=record['name'].removeprefix('memcheck-')
                if not record.get('zero_error_summary') or not re.search(r'ERROR SUMMARY: 0 errors',text) or MARKERS[stem] not in text:raise ValueError('Memory check lacks numerical/zero-error result')
                checks.append((record,folder))
            elif kind=='racecheck':
                if 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' not in text or 'TILED GEMM PASS: 3 cases, errors=0' not in text:raise ValueError('Racecheck failed')
                checks.append((record,folder))
            elif kind=='nsight_compute':
                if record['exit_code']==1:
                    if 'ERR_NVGPUCTRPERM' not in text:raise ValueError('Unexpected profiler failure')
                elif 'OVERLAP PASS:' not in text or 'Profiling "transform"' not in text:raise ValueError('Counter capture missing workload/kernel')
            elif kind=='nsight_compute_report_analysis':
                report=next(base.glob('gpu-checks/*/ncu-transform-privileged.ncu-rep'))
                if sha(report)!=record['report_sha256'] or 'transform' not in text:raise ValueError('Counter report differs')
                analyses.append((record,folder,report))
    if len(checks)!=5 or sum(r['kind']=='memcheck' for r,_ in checks)!=4 or len(analyses)!=1:raise ValueError('Expected four memory checks, one racecheck and one counter analysis')
    prefix='validation/'+DATASET+'/'
    def link(path,title):return '['+title+']('+prefix+path.relative_to(public).as_posix()+')'
    lines=['## Memory, race and counter checks','',
           'Four memory-check executions passed their numerical oracles and reported **zero errors**. The SIMT GEMM racecheck reported **zero hazards, zero errors and zero warnings**. These checks cover the captured executions, not every possible schedule.','',
           '| Check | Actual UTC | Captured result |','| --- | --- | --- |']
    for r,folder in checks:
        lines.append('| '+r['name']+' | `'+r['started_utc']+'` | '+link(folder/r['log'],'Actual output')+' |')
    for r,folder,report in analyses:
        lines+=['', 'A basic Nsight Compute capture of the selected `transform` kernel completed after a process-local privileged run. The original ordinary-user **ERR_NVGPUCTRPERM** diagnostic remains in the manifest/logs; no system-wide counter policy was changed. Counter replay changes timing, so the profiled application span is not an unprofiled performance result.', '',
                link(report,'Actual Compute report')+' · '+link(folder/r['log'],'Captured counter details')+'.']
    return '\n'.join(lines)+'\n'

def summary(rows,system,profiles,public):
    prefix='validation/'+DATASET+'/'
    lines=['# H100 CuTe/CUTLASS workshop execution — November 18, 2026','',
           '**Eight CUDA translation units compiled, two Python references byte-compiled, and all eleven workload invocations passed.** The GPU programs include a CuTe copy, a SIMT GEMM, two CUTLASS GEMMs and the serial/two-stream exercise. Three CUDA-compiled CuTe programs and both Python references execute CPU checks; they are not GPU-kernel benchmarks.','',
           'System: **1× NVIDIA H100 PCIe, compute capability 9.0, 81559 MiB**, driver **580.105.08**, CUDA compiler **12.8.93**, GCC **13.3.0**. CUTLASS **v4.3.6**, commit `'+system['cutlass_sha']+'`. All execution dates below are actual UTC stage timestamps.','',
           f'Environment: **Ubuntu 24.04.5 LTS**, Linux **6.8.0-1064-nvidia**, Python **3.12.3**; Nsight Systems **2024.6.2.225**, Nsight Compute **2025.1.1.0**, Compute Sanitizer **2025.1.0.0**. [Captured OS]({prefix}system-9.log) · [Systems version]({prefix}system-4.log) · [Compute version]({prefix}system-5.log) · [Sanitizer version]({prefix}system-6.log).','',
           f'[System metadata]({prefix}system.json) · [GPU inventory]({prefix}system-2.log) · [Results manifest]({prefix}results.json) · [Integrity inventory]({prefix}integrity.json).','',
           'The captured book checkout base and working-tree state are retained in metadata; each result is tied to its exact captured source hash. Earlier compilation/packaging failures and supplemental checks are recorded separately when linked below. These results do not establish B200 compatibility, native NVFP4, or an application speedup.','',
           '| Program | Scope | Actual run UTC | Result |','| --- | --- | --- | --- |']
    for r in rows:
        name=Path(r['source']).name;stem=Path(name).stem
        for run in r['runs']:
            mode=' '+run['command'][-1] if stem=='nsys_overlap' else ''
            lines += [f"| [{name+mode}]({PAGES[stem]}) | {r['kind'].replace('_',' ')} | `{run['started_utc']}` | PASS |"]
    lines+=['','## Actual capture tools and reports','',
            'The two Nsight Systems captures contain the passing workload and a captured `transform` kernel summary. Capture completion is not evidence of useful overlap or speedup; inspect the timeline and compare the stated measurement boundary.','']
    for p in profiles:
        mode='serial' if any(a=='serial' for a in p['command']) else 'overlap'
        report='timeline-'+mode+'.nsys-rep'
        lines += [f"- **{mode}**: capture UTC `{p['started_utc']}`. [Capture log]({prefix}{p['log']}) · [Kernel summary]({prefix}{p['stats']['log']}) · [Report]({prefix}{report})."]
    extra=supplemental(public)
    if extra:lines+=['',extra]
    lines+=['','## Additional introductory stride exercise','',
            'The separate [Chapter 16 stride oracle](16-cute-strides.md) was byte-compiled and executed locally on November 18, 2026. Its [commands and timestamps](validation/2026-11-18-strides/commands.json), [system information](validation/2026-11-18-strides/system.json) and [output](validation/2026-11-18-strides/run.log) identify a CPU Python address model, not another H100 workload. The ten-source H100 dataset above remains unchanged.','',
            '## Reproduce the run','',
            'Follow [the introductory workspace](15-cute-layouts.md) and [the complete test plan](examples/advanced/README.md). The collector requires a clean pinned CUTLASS dependency, a fresh output directory and explicit `--execute --profiles`. It never treats a host model as a numerical GPU workload.','']
    return '\n'.join(lines)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('capture',type=Path);ap.add_argument('--refresh',action='store_true');args=ap.parse_args()
    capture=args.capture.resolve();rows,system,profiles=validate(capture)
    public=ROOT/'docs/validation'/DATASET
    if not args.refresh:
        public.mkdir(parents=True,exist_ok=False)
        binaries={Path(r['source']).stem for r in rows}
        for p in capture.iterdir():
            if p.is_file() and p.name not in binaries and p.suffix not in {'.qdstrm','.sqlite'}:
                shutil.copy2(p,public/('capture-integrity.json' if p.name=='integrity.json' else p.name))
    validate(public)
    for row in rows:
        stem=Path(row['source']).stem;page=ROOT/'docs'/PAGES[stem];text=page.read_text();new=section(row,public)
        pattern=re.compile(r'<!-- workshop:start:'+re.escape(stem)+r' -->.*?<!-- workshop:end:'+re.escape(stem)+r' -->',re.S)
        text=pattern.sub(lambda _:new,text) if pattern.search(text) else text.rstrip()+'\n\n'+new+'\n'
        page.write_text(text)
    page=ROOT/'docs/cutlass-workshop-validation.md';page.write_text(summary(rows,system,profiles,public))
    inventory={p.relative_to(public).as_posix():sha(p) for p in public.rglob('*') if p.is_file() and p!=public/'integrity.json'}
    (public/'integrity.json').write_text(json.dumps({'dataset':DATASET,'files':inventory,'summary_sha256':sha(page)},indent=2)+'\n')
    print('Published exact workshop records:',len(rows),'sources;',sum(len(r['runs']) for r in rows),'workloads.')
if __name__=='__main__':main()
