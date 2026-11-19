from pathlib import Path
import datetime as dt,hashlib,json,os,platform,shutil,subprocess,sys
base=Path.home()/'book-attention-20261118';out=base/'artifacts/online-row-01';out.mkdir(parents=True,exist_ok=False)
for name in ['attention_online.cu','attention_walkthrough.py']:shutil.copy2(base/'source'/name,out/name)
shutil.copy2(__file__,out/'collector.py')
now=lambda:dt.datetime.now(dt.timezone.utc).isoformat()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1')
report={'started_utc':now(),'scope':'Chapter30 FP32 CUDA online attention and independent binary64 output/LSE checks; Python mathematical oracle rerun.','system':{'platform':platform.platform(),'python':sys.version,'machine':platform.machine(),'CUDA_VISIBLE_DEVICES':'1'},'commands':[],'status':'INCOMPLETE'}
def save():
 report['finished_utc']=now();(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
def run(label,cmd,timeout=180):
 start=now();r=subprocess.run(cmd,cwd=out,env=env,text=True,capture_output=True,timeout=timeout)
 (out/(label+'.stdout.log')).write_text(r.stdout);(out/(label+'.stderr.log')).write_text(r.stderr)
 report['commands'].append({'label':label,'argv':cmd,'started_utc':start,'finished_utc':now(),'exit_code':r.returncode,'stdout':label+'.stdout.log','stderr':label+'.stderr.log'})
 save()
 if r.returncode:raise RuntimeError(label+' failed: '+r.stderr)
 return r.stdout+r.stderr
try:
 for label,cmd in [('gpu-system',['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap','--format=csv']),('topology',['nvidia-smi','topo','-m']),('nvcc',['nvcc','--version']),('host-compiler',['g++','--version']),('os',['cat','/etc/os-release'])]:run(label,cmd)
 run('compile',['nvcc','-std=c++17','-O2','-arch=sm_90a','attention_online.cu','-o','attention_online'])
 result=run('execute',['./attention_online'])
 if 'CUDA ATTENTION PASS: seven cases' not in result:raise RuntimeError('Missing CUDA numerical PASS')
 result=run('memcheck',['compute-sanitizer','--tool','memcheck','--error-exitcode','3','./attention_online'])
 if 'ERROR SUMMARY: 0 errors' not in result:raise RuntimeError('Missing clean memcheck summary')
 run('python-compile',['python3','-m','py_compile','attention_walkthrough.py'])
 result=run('python-execute',['python3','attention_walkthrough.py'])
 if 'ATTENTION WALKTHROUGH PASS' not in result:raise RuntimeError('Missing Python PASS')
 report['status']='PASS';save()
except Exception as exc:
 report.update(status='FAIL',error=str(exc));save();raise
finally:
 files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='attention_online'}
 (out/'integrity.json').write_text(json.dumps(files,indent=2)+'\n')
print(report['status'],out)
