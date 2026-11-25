"""Regressions for state corruption and unsupported equal-slot placement."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/examples/inference-systems'))
import numpy as np
from ch43_paged_kv_cache import BlockAllocator, PagedKVCache
from ch33_moe_expert_parallel import eplb_place
import torch


def rejects(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('invalid input accepted')


rejects(lambda:BlockAllocator(-1))
a=BlockAllocator(2); a.alloc();a.alloc()
rejects(lambda:a.incref(-1));rejects(lambda:a.decref(-1))
assert a.ref==[1,1]
c=PagedKVCache(3,4,2,8);c.add_seq('p')
rejects(lambda:c.append('p',np.zeros((1,8)),np.zeros((2,8))))
assert c.used_blocks()==0 and c.seqs['p']['table']==[]
c.append('p',np.ones((2,8)),np.ones((2,8)));c.fork('p','child')
rejects(lambda:c.append('child',np.zeros((2,8)),np.zeros((3,8))))
assert c.used_blocks()==1 and c.alloc.ref[0]==2
c.check_invariants()
rejects(lambda:eplb_place(torch.ones(5),2,0))
print('PASS invalid K/V append preserves allocator and fork state; negative indices and unequal-slot placement rejected')

from ch36_benchmark_teardown import Rec, summarize
r=Rec(intended=0,sent=0,first=1,end=1.1,ntok=1,ok=True)
assert summarize([r],warmup=0)['tpot_p50']==0
print('PASS one-token response has zero inter-token intervals')
import asyncio
from ch36_benchmark_teardown import Target, one_request


async def malformed_stream():
    async def handler(reader, writer):
        await reader.readuntil(b'\r\n\r\n')
        writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n'
                     b'data: invalid-json\n\n'
                     b'data: {"choices":[{"text":"x"}],"usage":{"completion_tokens":1}}\n\n'
                     b'data: [DONE]\n\n')
        await writer.drain()
        writer.close()
    server=await asyncio.start_server(handler,'127.0.0.1',0)
    try:
        port=server.sockets[0].getsockname()[1]
        rec=Rec(intended=0)
        await one_request(Target(port=port),1,1,rec)
        assert not rec.ok and rec.err
    finally:
        server.close();await server.wait_closed()


asyncio.run(malformed_stream())
print('PASS malformed stream is rejected even when a later event supplies valid usage and DONE')
