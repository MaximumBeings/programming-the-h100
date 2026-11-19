from pathlib import Path
import importlib.util,json,os,hashlib,shutil,sys
base=Path.home()/'book-attention-20261118';out=base/'artifacts/attention-memory-01';out.mkdir(parents=True)
spec=importlib.util.spec_from_file_location('collector',base/'tools/run_cutlass_attention.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
shutil.copy2(__file__,out/'collector.py')
r={'scope':'Instrumented memory checks, not performance measurements; CUDA-visible device identity probes.','commands':[],'status':'INCOMPLETE'}
def save():(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
probe='''import ctypes,json,os
lib=ctypes.CDLL("libcuda.so.1")
def ck(n):
 if n: raise RuntimeError("CUDA Driver error "+str(n))
ck(lib.cuInit(0));dev=ctypes.c_int();ck(lib.cuDeviceGet(ctypes.byref(dev),0));name=ctypes.create_string_buffer(128);ck(lib.cuDeviceGetName(name,128,dev));u=(ctypes.c_ubyte*16)();ck(getattr(lib,"cuDeviceGetUuid_v2",lib.cuDeviceGetUuid)(ctypes.byref(u),dev));h=bytes(u).hex();print(json.dumps({"CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"],"name":name.value.decode(),"uuid":"GPU-"+h[:8]+"-"+h[8:12]+"-"+h[12:16]+"-"+h[16:20]+"-"+h[20:]}))
'''
try:
 for visible in ['0','1']:
  os.environ['CUDA_VISIBLE_DEVICES']=visible;rec,text=m.record([sys.executable,'-c',probe],out,'device-'+visible,30);rec['CUDA_VISIBLE_DEVICES']=visible;r['commands'].append(rec);save();assert rec['exit_code']==0,text
 os.environ['CUDA_VISIBLE_DEVICES']='0'
 _,_,cases=m.commands(base/'artifacts/hopper-fmha',4,3,True)
 for name,cmd,count in cases:
  if name not in ['d64-full','d64-backward-no']:continue
  args=['compute-sanitizer','--tool','memcheck','--error-exitcode','3']+cmd
  rec,text=m.record(args,out,name+'-memcheck',900);rec['status']=m.classify(rec['exit_code'],text,count,cmd);rec['CUDA_VISIBLE_DEVICES']='0';r['commands'].append(rec);save();assert rec['status']=='PASS' and 'ERROR SUMMARY: 0 errors' in text,text
 r['status']='PASS';save()
except Exception as e:r.update(status='FAIL',error=str(e));save();raise
finally:
 (out/'integrity.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2)+'\n')
print(r['status'],out)
