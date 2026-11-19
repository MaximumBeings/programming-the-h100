from pathlib import Path
import os,sys,json,subprocess,hashlib,datetime,shlex,signal,time,re
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).resolve().parent
out=root/'results';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def capture(args,path,timeout=40):
 start=now();t=time.monotonic()
 with path.open('w') as f:
  f.write('$ '+shlex.join(args)+'\n');f.flush()
  p=subprocess.Popen(args,cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  timed=False
  try:code=p.wait(timeout)
  except subprocess.TimeoutExpired:
   timed=True;os.killpg(p.pid,signal.SIGKILL);p.wait();code=p.returncode
 return dict(command=args,started_utc=start,finished_utc=now(),exit_code=code,timed_out=timed,seconds=time.monotonic()-t,log=str(path.relative_to(root)),log_sha256=sha(path))
meta=[]
for label,args in [('gpus',['nvidia-smi','--query-gpu=name,uuid,compute_cap,driver_version,memory.total','--format=csv']),('nvcc',['nvcc','--version']),('topology',['nvidia-smi','topo','-m']),('os',['cat','/etc/os-release'])]:meta.append(capture(args,out/(label+'.log'),10))
(out/'system.json').write_text(json.dumps(meta,indent=2)+'\n')
jobs=[]
for row in json.loads((root/'original-programs.json').read_text()):jobs.append(dict(source='original/'+row['example'],kind='original',page=row['page']))
for p in sorted((root/'docs/examples/advanced').glob('*.cu')):jobs.append(dict(source=str(p.relative_to(root)),kind='workshop'))
for p in sorted((root/'docs/examples/advanced').glob('*.py')):jobs.append(dict(source=str(p.relative_to(root)),kind='cpu_reference'))
jobs.append(dict(source='docs/examples/attention/attention_online.cu',kind='attention_reference'))
jobs.append(dict(source='docs/examples/attention/88_hopper_fmha/88_hopper_fmha.cu',kind='hopper_fmha'))
def test(job):
 source=root/job['source'];folder=out/source.stem;folder.mkdir();(folder/source.name).write_bytes(source.read_bytes())
 row=dict(job,sha256=sha(source),runs=[]);binary=folder/'example';text=source.read_text();arch='sm_100'
 if source.suffix=='.py':cmd=['python3','-m','py_compile',str(source)]
 else:
  cmd=['nvcc','-std=c++17','-O2','-arch='+arch,'--expt-relaxed-constexpr','-I'+str(root/'cutlass/include'),'-I'+str(root/'cutlass/tools/util/include'),str(source),'-o',str(binary)]
  if 'cudaLaunchDevice' in text or 'cudaGridDependencySynchronize' in text:cmd+=['-rdc=true','-lcudadevrt']
  if 'cuTensorMapEncode' in text or job['kind']=='hopper_fmha' or source.stem.startswith('cutlass_gemm'):cmd+=['-lcuda']
  if '#include <nccl.h>' in text:cmd+=['-lnccl']
  if 'cublas' in text:cmd+=['-lcublas','-Xcompiler=-pthread']
 row['compile']=capture(cmd,folder/'compile.log',45 if job['kind']=='hopper_fmha' else 35)
 if row['compile']['exit_code']!=0 and ('wgmma' in source.stem or 'setmaxnreg' in source.stem) and not row['compile']['timed_out']:
  cmd=[x.replace('-arch=sm_100','-arch=sm_90a') for x in cmd];row['hopper_compile']=capture(cmd,folder/'compile-hopper.log',25)
  if row['hopper_compile']['exit_code']==0:row['binary_target']='sm_90a'
  else:row['status']='COMPILE_UNSUPPORTED_OR_FAILED';return row
 elif row['compile']['exit_code']!=0:row['status']='COMPILE_TIMEOUT' if row['compile']['timed_out'] else 'COMPILE_UNSUPPORTED_OR_FAILED';return row
 else:row['binary_target']='cpu' if source.suffix=='.py' else arch
 argslist=[['serial'],['overlap']] if source.stem=='nsys_overlap' else [['--b=1','--h=2','--q=128','--k=128','--d=64','--iterations=1']] if job['kind']=='hopper_fmha' else [[]]
 for i,args in enumerate(argslist):
  command=(['python3',str(source)] if source.suffix=='.py' else [str(binary)])+args
  run=capture(command,folder/f'run-{i}.log',30);output=(root/run['log']).read_text()
  if re.search(r'Requires Hopper (?:SM|compute)|only (?:supported|supports).*Hopper|requires.*Hopper.*9\.0|only runs on.*Hopper|Hopper.*only',output,re.I):run['status']='ARCHITECTURE_GUARD'
  elif run['exit_code']==0 and not re.search(r'\b(?:FAIL|ERROR)\b',output):run['status']='COMPLETED'
  else:run['status']='TIMEOUT' if run['timed_out'] else 'RUNTIME_UNSUPPORTED_OR_FAILED'
  row['runs'].append(run)
 row['status']='COMPLETED' if all(r['status']=='COMPLETED' for r in row['runs']) else row['runs'][0]['status']
 (folder/'record.json').write_text(json.dumps(row,indent=2)+'\n');print(row['status'],source.name,flush=True);return row
records=[]
with ThreadPoolExecutor(max_workers=5) as pool:
 for row in pool.map(test,jobs):records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
from collections import Counter
(out/'summary.json').write_text(json.dumps(dict(Counter(r['status'] for r in records)),indent=2)+'\n');print((out/'summary.json').read_text(),flush=True)
files={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and ('cutlass' not in p.relative_to(root).parts) and '__pycache__' not in p.parts and p.suffix in {'.cu','.cpp','.py','.log','.json'}}
(root/'integrity.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
import tarfile
with tarfile.open('/home/ubuntu/h100-book-b200-20261119-evidence.tgz','w:gz') as tar:
 for name in list(files)+['integrity.json']:tar.add(root/name,arcname=name,recursive=False)
(root/'DONE').write_text(now());print('EVIDENCE_READY',flush=True)
