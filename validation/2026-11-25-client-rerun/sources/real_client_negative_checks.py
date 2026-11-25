"""Expected HTTP error and timeout against the same live vLLM server."""
import asyncio
from dataclasses import asdict
import json
import math
import time
from ch36_benchmark_teardown import Rec, Target, one_request


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


async def main():
    records=[]
    for name,target,timeout,nout in [
        ('unknown_model',Target(model='nonexistent-validation-model'),10.0,16),
        ('client_deadline',Target(model='Qwen/Qwen2.5-0.5B-Instruct'),0.005,256)]:
        rec=Rec(intended=time.perf_counter())
        await one_request(target,128,nout,rec,timeout=timeout)
        assert not rec.ok and rec.err, (name,rec)
        assert math.isfinite(rec.end) and rec.end>=rec.intended,(name,rec)
        if name=='unknown_model':assert rec.err.startswith('HTTP/'),rec
        else:assert rec.err=='timeout',rec
        records.append(dict(case=name,expected_failure=True,passed=True,record=clean(asdict(rec))))
    print(json.dumps(records,indent=2,allow_nan=False))


if __name__=='__main__':asyncio.run(main())
