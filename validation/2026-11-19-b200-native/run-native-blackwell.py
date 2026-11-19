from pathlib import Path
import os,json,subprocess,hashlib,datetime,signal,shlex,re,tarfile
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).resolve().parent;out=root/'results';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def capture(name,args,timeout=90):
 pth=out/(name+'.log');start=utc()
 with pth.open('w') as f:
  f.write('$ '+shlex.join(args)+'\n');f.flush();p=subprocess.Popen(args,cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  timed=False
  try:code=p.wait(timeout)
  except subprocess.TimeoutExpired:timed=True;os.killpg(p.pid,signal.SIGKILL);p.wait();code=p.returncode
 return dict(name=name,command=args,started_utc=start,finished_utc=utc(),exit_code=code,timed_out=timed,log=pth.name,log_sha256=sha(pth))
headers=['-I'+str(root/'cutlass/include'),'-I'+str(root/'cutlass/tools/util/include')]
jobs=[('blackwell_gemm',root/'docs/examples/blackwell/blackwell_gemm.cu'),('blackwell_gemm_relu',root/'docs/examples/blackwell/blackwell_gemm_relu.cu'),('blackwell_fmha',root/'cutlass/examples/77_blackwell_fmha/77_blackwell_fmha.cu')]
def build(job):
 name,source=job;cmd=['nvcc','-std=c++17','-O2','-arch=sm_100a','--expt-relaxed-constexpr']+headers
 if name=='blackwell_fmha':cmd+=['-DFP16','--use_fast_math','-I'+str(source.parent)]
 cmd+=[str(source),'-o',str(root/name),'-lcuda'];record=capture(name+'-compile',cmd,100);record.update(source=str(source.relative_to(root)),source_sha256=sha(source));print(name,'compile',record['exit_code'],flush=True);return record
builds=list(ThreadPoolExecutor(max_workers=3).map(build,jobs));(out/'builds.json').write_text(json.dumps(builds,indent=2)+'\n')
runs=[]
for build in builds:
 if build['exit_code']!=0:print((out/build['log']).read_text()[-6000:],flush=True);continue
 name=build['name'].removesuffix('-compile')
 if name!='blackwell_fmha':cases=[(name,[])]
 else:cases=[('fmha-full-d64',['--b=1','--h=2','--q=128','--k=128','--d=64','--mask=no']),('fmha-causal-d64',['--b=1','--h=2','--q=128','--k=128','--d=64','--mask=causal']),('fmha-gqa-d64',['--b=1','--h=4','--h_k=2','--q=128','--k=128','--d=64','--mask=no']),('fmha-full-d128',['--b=1','--h=2','--q=128','--k=128','--d=128','--mask=no'])]
 for label,args in cases:
  if name=='blackwell_fmha':args+=['--verify','--iterations=1','--warmup_iterations=0','--verbose']
  rec=capture(label,[str(root/name)]+args,30);text=(out/rec['log']).read_text();rec['verified_variants']=len(re.findall(r'^\s*\[OK\]',text,re.M))
  good=rec['verified_variants']>0 and not re.search(r'\[FAIL\]|\[--\]|failed O:|failed LSE:|Reference check failed|not supported|requires a GPU|[Nn]an|[Ii]nf TFLOPS',text) if name=='blackwell_fmha' else len(re.findall(r'^(?:FUSED )?GEMM CASE PASS ',text,re.M))==3 and ('GEMM PASS: 3 full CPU-reference cases' in text or 'FUSED GEMM PASS: 3 full CPU-reference cases' in text)
  rec['status']='PASS' if rec['exit_code']==0 and good else 'FAIL';runs.append(rec);print(label,rec['status'],text[-1800:],flush=True)
(out/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
for name,args in [('gpu',['nvidia-smi','--query-gpu=name,uuid,compute_cap,driver_version,memory.total','--format=csv']),('nvcc',['nvcc','--version']),('os',['cat','/etc/os-release'])]:capture(name,args,10)
files={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and 'cutlass' not in p.relative_to(root).parts and '__pycache__' not in p.parts and p.suffix in {'.json','.log','.cu','.py'}}
# Pin every upstream input, without copying dependency binaries into the evidence.
upstream={p.relative_to(root/'cutlass').as_posix():sha(p) for p in (root/'cutlass').rglob('*') if p.is_file()};(out/'cutlass-input-hashes.json').write_text(json.dumps(upstream,indent=2)+'\n');files['results/cutlass-input-hashes.json']=sha(out/'cutlass-input-hashes.json')
(root/'integrity.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
with tarfile.open('/home/ubuntu/native-blackwell-20261119-evidence.tgz','w:gz') as tar:
 for name in list(files)+['integrity.json']:tar.add(root/name,arcname=name,recursive=False)
(root/'DONE').write_text(utc());print('NATIVE_EVIDENCE_READY',flush=True)
