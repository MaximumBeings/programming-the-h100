import datetime,hashlib,json,os,pathlib,subprocess,time
out=pathlib.Path('/tmp/h100-supplement-20261121');out.mkdir(exist_ok=False)
rows=[]
def cap(name,cmd,env=None,timeout=90):
 started=datetime.datetime.now(datetime.timezone.utc).isoformat(); t=time.monotonic()
 try:
  r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,timeout=timeout);code=r.returncode;data=r.stdout
 except subprocess.TimeoutExpired as e:code='timeout';data=e.stdout or b''
 (out/(name+'.log')).write_bytes(data)
 row=dict(name=name,command=cmd,started_utc=started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-t,exit_code=code,log=name+'.log',log_sha256=hashlib.sha256(data).hexdigest());rows.append(row);(out/'manifest.json').write_text(json.dumps(rows,indent=2));print(name,code,flush=True)
cap('system',['bash','-lc','date -u; uname -a; nvidia-smi -q; nvidia-smi topo -m; nvcc --version; python3 -m pip freeze'])
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1';env.pop('TRITON_INTERPRET',None)
p=pathlib.Path('/tmp/triton-review/docs/examples/gpu_smoke.py');(out/'gpu_smoke.py').write_bytes(p.read_bytes())
cap('triton-smoke',['python3',str(p)],env)
cap('two-gpu-examples',['python3','/tmp/h100-review/tools/validate_examples.py','ex11_nvlink_bandwidth','ex12_nccl_allreduce','--out','/tmp/h100-two-gpu-20261121'],timeout=180)
root=pathlib.Path('/tmp/h100-rerun-20261121')
for chapter,example in [('02-clusters-inline-ptx','ex01_cluster_inspect'),('02-clusters-inline-ptx','ex02_ptx_basics'),('02-clusters-inline-ptx','ex06_comprehensive'),('05-cp-async-bulk-l2-hinting','ex02_reduce_async')]:
 binary=root/chapter/example/'example'
 for tool in ['memcheck','racecheck','synccheck']:
  cap(example+'-'+tool,['compute-sanitizer','--tool',tool,'--error-exitcode','91',str(binary)],timeout=60)
cap('attention-memcheck',['compute-sanitizer','--tool','memcheck','--error-exitcode','91','/tmp/h100-attention-20261121/attention'])
