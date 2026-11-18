from pathlib import Path
from datetime import datetime,timezone
import ctypes,json,re
header=Path('/usr/include/cuda.h').read_text();driver=ctypes.CDLL('libcuda.so.1')
def check(code):
 if code:raise RuntimeError('CUDA Driver API error '+str(code))
check(driver.cuInit(0));device=ctypes.c_int();check(driver.cuDeviceGet(ctypes.byref(device),0));data={'utc':datetime.now(timezone.utc).isoformat(),'device_ordinal':0,'header':'/usr/include/cuda.h','attributes':{}}
for name in ['MULTIPROCESSOR_COUNT','ASYNC_ENGINE_COUNT','MAX_THREADS_PER_MULTIPROCESSOR','MAX_SHARED_MEMORY_PER_MULTIPROCESSOR','MAX_SHARED_MEMORY_PER_BLOCK_OPTIN']:
 match=re.search(r'\bCU_DEVICE_ATTRIBUTE_'+name+r'\s*=\s*(\d+)',header)
 if not match:raise RuntimeError('Header attribute not found: '+name)
 value=ctypes.c_int();check(driver.cuDeviceGetAttribute(ctypes.byref(value),int(match[1]),device));data['attributes'][name]=value.value
print(json.dumps(data,indent=2))
