#!/usr/bin/env python3
"""Capture one standalone GPU audit with an explicit architecture and success marker."""
import argparse,datetime,hashlib,json,pathlib,platform,shutil,subprocess,sys,time

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def capture(command,folder,name,timeout):
    start=utc();clock=time.monotonic()
    try:
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',timeout=timeout)
        output=result.stdout;code=result.returncode;timed=False
    except subprocess.TimeoutExpired as e:
        output=e.stdout or '';output=output.decode(errors='replace') if isinstance(output,bytes) else output
        output+=f'\nTIMEOUT after {timeout}s\n';code=None;timed=True
    log=folder/(name+'.log');log.write_text(output)
    return dict(command=command,started_utc=start,finished_utc=utc(),seconds=time.monotonic()-clock,exit_code=code,timed_out=timed,log=log.name,log_sha256=sha(log))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--arch',required=True);p.add_argument('--pass-marker',required=True);a=p.parse_args()
    out=a.out.resolve()
    if out.exists() and any(out.iterdir()):p.error('Output must be fresh; existing evidence is never overwritten')
    out.mkdir(parents=True,exist_ok=True);source=out/a.source.name;shutil.copyfile(a.source,source)
    m=dict(timestamp_utc=utc(),source=source.name,source_sha256=sha(source),arch=a.arch,pass_marker=a.pass_marker,host_python=platform.python_version(),status='UNEXECUTED')
    probes=[['nvidia-smi','--query-gpu=name,uuid,compute_cap,driver_version,memory.total','--format=csv'],['nvidia-smi','topo','-m'],['nvcc','--version'],['gcc','--version'],['uname','-a']]
    text=[]
    for cmd in probes:
        r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30);text+=['$ '+' '.join(cmd),f'Exit: {r.returncode}',r.stdout]
    system=out/'system-info.log';system.write_text('\n'.join(text));m['system_sha256']=sha(system)
    binary=out/'example';command=['nvcc','-std=c++17','-arch='+a.arch,'-O2',str(source),'-o',str(binary),'-lnccl','-Xcompiler','-pthread']
    if 'cublas' in source.read_text():command+=['-lcublas']
    m['compile']=capture(command,out,'compile',180)
    if m['compile']['exit_code']==0:
        m['binary_sha256']=sha(binary);m['run']=capture([str(binary)],out,'run',180)
        m['status']='PASS' if m['run']['exit_code']==0 and a.pass_marker in (out/'run.log').read_text() else 'FAIL'
    else:m['status']='COMPILE_FAIL'
    m['finished_utc']=utc();(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m['status'],out,flush=True)
    return m['status']!='PASS'
if __name__=='__main__':sys.exit(main())
