import pathlib,subprocess,sys,time,json,datetime,urllib.request
root=pathlib.Path('/tmp/client-rerun');src=root/'sources'
for i in range(120):
 try:
  with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2) as r:
   assert r.status==200
   (root/'health.json').write_text(json.dumps(dict(status=r.status,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
  break
 except Exception:time.sleep(2)
else:raise RuntimeError('server did not become healthy')
model='Qwen/Qwen2.5-0.5B-Instruct'
jobs=[('probe',[sys.executable,'ch47_vllm_probe.py','--model',model,'--output',str(root/'probe.json')]),('open_loop',[sys.executable,'ch36_benchmark_teardown.py','run','--url','http://127.0.0.1:8000/v1/completions','--model',model,'--rate','2','--n','16','--warmup','2','--seed','1','--n-in','64','--n-out','16','--out',str(root/'open-loop.jsonl')]),('negative',[sys.executable,'real_client_negative_checks.py'])]
for name,cmd in jobs:
 start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 r=subprocess.run(cmd,cwd=src,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
 (root/(name+'.log')).write_bytes(r.stdout)
 commands=json.loads((root/'commands.json').read_text());commands[name]=dict(command=cmd,start_utc=start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=r.returncode,log=name+'.log');(root/'commands.json').write_text(json.dumps(commands,indent=2))
 print(name,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(name+' failed; raw output saved')
 if name=='open_loop':
  rows=[json.loads(x) for x in (root/'open-loop.jsonl').read_text().splitlines()];assert len(rows)==16 and all(x['ok'] for x in rows)
 if name=='negative':(root/'negative.json').write_bytes(r.stdout)
(root/'metrics.txt').write_bytes(urllib.request.urlopen('http://127.0.0.1:8000/metrics').read())
(root/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
(root/'os-release.txt').write_bytes(pathlib.Path('/etc/os-release').read_bytes())
(root/'suite.exit').write_text('0')
print('ALL REAL CLIENT CHECKS PASS',flush=True)
