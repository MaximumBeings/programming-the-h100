import datetime, hashlib, json, os, pathlib, platform, subprocess, sys, time
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);args=ap.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];out=args.output;out.mkdir(parents=True,exist_ok=True)
rows=[]
for p in sorted((root/'docs/examples/inference-systems').glob('*.py')):
    if p.stem.startswith('ch') and int(p.stem[2:4]) > 46:
        continue  # Chapter 47 needs explicit CUDA/server commands, not a CPU reference run.
    opts=['--test'] if p.stem[:4] in ['ch32','ch33','ch34','ch35'] else ['--selftest'] if p.stem[:4] in ['ch36','ch38','ch39','ch40'] else ['tests'] if p.stem[:4] in ['ch41','ch42','ch43','ch44','ch45','ch46'] else []
    for mode,args in [('tests',opts),('full',[])]:
        if mode=='tests' and not opts:continue
        cmd=[sys.executable,str(p),*args]; start=time.time()
        try:r=subprocess.run(cmd,cwd=p.parent,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=480);rc=r.returncode;data=r.stdout
        except subprocess.TimeoutExpired as e:rc=124;data=(e.stdout or b'')+b'\nRUNNER TIMEOUT\n'
        log=p.stem+'-'+mode+'.log';(out/log).write_bytes(data)
        rows.append(dict(source=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),command=cmd,scope='CPU reference or simulation; placement on GPU host does not imply CUDA use',mode=mode,exit_code=rc,seconds=time.time()-start,log=log))
        (out/'results.json').write_text(json.dumps(dict(date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),results=rows),indent=2))
        print(p.name,mode,rc,flush=True)
