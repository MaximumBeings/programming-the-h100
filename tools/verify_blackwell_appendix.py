"""Check captured native B200 results and complete appendix sources without a GPU."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,math,re,shlex,zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/validation/2026-11-19-b200-native'
COMPAT=ROOT/'docs/validation/2026-11-19-b200-compatibility'
PIN='1afc6d355b8cb992398f17beb4425702f1d57ddd'

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def inventory(base,name):
    files=json.loads((base/name).read_text())['files']
    actual={p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}-{name}
    require(set(files)==actual,'Raw Blackwell inventory differs')
    for name,digest in files.items():
        rel=Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts and sha(base/rel)==digest,'Changed Blackwell evidence: '+name)

def command(row,base,subdir='results',success=True):
    path=base/subdir/row['log']
    require(sha(path)==row['log_sha256'],'Blackwell command log hash differs')
    require(path.read_text().startswith('$ '+shlex.join(row['command'])+'\n'),'Blackwell command header differs')
    if success:require(row['exit_code']==0 and not row.get('timed_out',False),'Blackwell command did not complete')
    start,finish=(datetime.fromisoformat(row[k]) for k in ('started_utc','finished_utc'))
    require(start.utcoffset()==timezone.utc.utcoffset(start) and finish.utcoffset()==timezone.utc.utcoffset(finish)
            and row['started_utc'].startswith('2026-11-19T') and start<=finish,'Invalid Blackwell execution dates')
    return path.read_text().split('\n',1)[1].rstrip()

def attention_verified(exit_code,text,d,kv,mask):
    header=f'###### B 1 H 2 H_K {kv} Q 256 K 256 D {d} Forward {mask} #SM 148'
    good=re.findall(r'^\s*\[OK\]\s+book sm100 FP16 forward\s*:\s*(\S+) TFLOPS/s\s*$',text,re.M)
    if exit_code!=0 or header not in text or len(good)!=1:return False
    if re.search(r'\[FAIL\]|\[--\]|failed O:|failed LSE:|Reference check failed|requires a GPU|[Nn]ot supported',text):return False
    try:return math.isfinite(float(good[0])) and float(good[0])>=0
    except ValueError:return False

def verify():
    inventory(BASE,'evidence-integrity.json');inventory(COMPAT,'integrity.json')
    page=(ROOT/'docs/appendix-blackwell-b200.md').read_text()
    names=['blackwell_gemm.cu','blackwell_gemm_relu.cu','blackwell_fmha_forward.cu']
    for name in names:
        source=ROOT/'docs/examples/blackwell'/name
        require(source.read_bytes()==(BASE/'docs/examples/blackwell'/name).read_bytes(),'Current native source differs from captured source')
        match=re.search(r'<!-- blackwell-source:'+re.escape(name)+r':start -->\n```cuda\n(.*?)```\n<!-- blackwell-source:'+re.escape(name)+r':end -->',page,re.S)
        require(match is not None and match[1]==source.read_text(),'Appendix omits/changes complete native source: '+name)
        require(sha(source) in page,'Appendix omits native source identity')
    builds=json.loads((BASE/'results/builds.json').read_text());basic=next(r for r in builds if r['name']=='blackwell_gemm-compile')
    require(basic['source_sha256']==sha(ROOT/'docs/examples/blackwell/blackwell_gemm.cu'),'GEMM build source differs')
    command(basic,BASE)
    fused=json.loads((BASE/'results/relu-fixed-results.json').read_text())
    require(fused['source_sha256']==sha(ROOT/'docs/examples/blackwell/blackwell_gemm_relu.cu'),'Fused build source differs')
    require(len(fused['records'])==2,'Incomplete fused build/run record')
    command(fused['records'][0],BASE)
    basicrun=next(r for r in json.loads((BASE/'results/runs.json').read_text()) if r['name']=='blackwell_gemm')
    for row,label in [(basicrun,'GEMM'),(fused['records'][1],'FUSED GEMM')]:
        text=command(row,BASE)
        require(text in page and row['started_utc'] in page,'Appendix changes GEMM output/date')
        require(len(re.findall(r'^'+label+r' CASE PASS .* errors=0$',text,re.M))==3
                and label+' PASS: 3 full CPU-reference cases' in text,'Incomplete native GEMM oracle')
        require(all(s in text for s in ['M=128 N=128 K=64 L=1','M=256 N=128 K=128 L=2','M=136 N=264 K=72 L=1']),'Native GEMM case differs')
    forward=json.loads((BASE/'results/forward-book-results.json').read_text());require(len(forward)==5,'Expected four native attention runs')
    command(forward[0],BASE)
    for row,(d,kv,mask) in zip(forward[1:],[(64,2,'None'),(64,2,'Causal'),(64,1,'None'),(128,2,'None')]):
        text=command(row,BASE)
        require('--verify' in row['command'] and attention_verified(row['exit_code'],text,d,kv,mask),'Native attention reference check differs')
        require(row['started_utc'] in page,'Appendix omits native attention date')
        if row is forward[1]:require(text in page,'Appendix changes native attention sample output')
    for row in [basic,fused['records'][0],forward[0]]:require('-arch=sm_100a' in row['command'],'Native program used wrong architecture')
    # Match the support bundle to the exact compiler inputs captured on the host.
    vendor=ROOT/'docs/examples/blackwell/77_blackwell_fmha';manifest=json.loads((vendor/'source-manifest.json').read_text())
    require(manifest['cutlass_commit']==PIN,'Wrong CUTLASS pin')
    require(set(manifest['files'])=={p.relative_to(vendor).as_posix() for p in vendor.rglob('*') if p.is_file()}-{'source-manifest.json'},'Upstream source inventory differs')
    inputs=json.loads((BASE/'results/final-cutlass-input-hashes.json').read_text())
    for name,digest in manifest['files'].items():
        original='LICENSE.txt' if name=='LICENSE.upstream.txt' else 'examples/77_blackwell_fmha/'+('README.md' if name=='README.upstream.txt' else name)
        require(sha(vendor/name)==digest and inputs.get(original)==digest,'Upstream support differs from tested inputs: '+name)
    upstream=(vendor/'77_blackwell_fmha.cu').read_text()
    require((ROOT/'docs/examples/blackwell/blackwell_fmha_forward.cu').read_text().startswith(upstream[:upstream.index('int main_single(')]),'Native attention core/reference differs from upstream')
    with zipfile.ZipFile(ROOT/'docs/examples/blackwell/blackwell-appendix-sources.zip') as archive:
        expected={p.relative_to(ROOT/'docs/examples/blackwell').as_posix():p for p in (ROOT/'docs/examples/blackwell').rglob('*') if p.is_file() and p.suffix!='.zip'}
        require(set(archive.namelist())==set(expected),'Blackwell source ZIP inventory differs')
        for name,path in expected.items():require(archive.read(name)==path.read_bytes(),'Blackwell source ZIP member differs')
    rows=json.loads((COMPAT/'results/results.json').read_text());require(len(rows)==41,'Compatibility source count differs')
    import validate_examples
    current={x['example']:x['sha256'] for x in validate_examples.programs()}
    require(len(current)==27,'Original program inventory drift')
    for row in rows:
        require(sha(COMPAT/row['source'])==row['sha256'],'Compatibility source snapshot changed')
        name=Path(row['source']).name
        if row['kind']=='original':require(current[name]==row['sha256'],'Compatibility original source differs')
        elif row['kind']!='hopper_fmha':require(sha(ROOT/row['source'])==row['sha256'],'Compatibility workshop/reference source differs')
    extra=[dict(row,log=row['stage']+'.log') for row in json.loads((COMPAT/'setmaxnreg-blackwell/results.json').read_text())];require(len(extra)==2,'Missing register-resizing rebuild')
    for row in extra:command(row,COMPAT,'setmaxnreg-blackwell')
    require('-arch=sm_100a' in extra[0]['command'] and 'ALL TESTS PASSED' in command(extra[1],COMPAT,'setmaxnreg-blackwell'),'Register-resizing Blackwell run failed')
    rawguard=(COMPAT/'results/88_hopper_fmha/run-0.log').read_text()
    require('requires a GPU' in rawguard and 'Hopper' in rawguard,'Hopper attention guard changed')
    print('PASS: complete native B200 sources, six CPU-oracle GEMM cases, four verified attention cases, pinned support ZIP and compatibility evidence.')

if __name__=='__main__':verify()
