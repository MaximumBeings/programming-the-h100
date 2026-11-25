"""Adversarial CPU regressions for the inference chapter review.

CPU only. MoE lifecycle tests require PyTorch; no GPU or model download.
Socket tests use temporary loopback listeners.
"""
import asyncio
import math
import sys
import time
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/examples/inference-systems'))
import ch32_inference_harness as c32
import ch34_kv_offload as c34
import ch35_disaggregation as c35
import ch36_benchmark_teardown as c36
import ch37_spec_decode_70b as c37


def injected_worker_failure(*args):
    raise RuntimeError("injected worker failure before publishing a result")


def injected_worker_stall(*args):
    time.sleep(10)


class ReviewTests(unittest.TestCase):
    def test_moe_argument_validation(self):
        import ch33_moe_expert_parallel as c33
        for kwargs in [dict(world=0),dict(world=3,n_experts=16),dict(topk=0),dict(n_tokens=0),
                       dict(worker_timeout=0),dict(mode='unsupported')]:
            with self.assertRaises(ValueError):c33.check_alltoall(**kwargs)

    def test_moe_failure_and_deadline_cleanup(self):
        import ch33_moe_expert_parallel as c33
        import multiprocessing as mp
        from unittest.mock import patch
        before={p.pid for p in mp.active_children()}
        with patch.object(c33,'_worker',injected_worker_failure):
            with self.assertRaises(RuntimeError):
                c33.check_alltoall(world=1,n_experts=1,topk=1,worker_timeout=15)
        with patch.object(c33,'_worker',injected_worker_stall):
            with self.assertRaises(TimeoutError):
                c33.check_alltoall(world=1,n_experts=1,topk=1,worker_timeout=.1)
        self.assertFalse({p.pid for p in mp.active_children()}-before)

    def test_mid_iteration_arrival_occupancy(self):
        reqs=[c32.Req(0,0,1,1),c32.Req(1,0.5,1,1)]
        cost=c32.CostModel(base=1,per_seq=0,per_kv_token=0,prefill_tps=math.inf)
        rows,_,area=c32.simulate(reqs,cost=cost)
        self.assertEqual(area,2.5)
        self.assertEqual(area,sum(r.finish-r.arrival for r in rows))

    def test_no_progress_configuration_rejected(self):
        with self.assertRaises(ValueError):c32.simulate([c32.Req(0,0,1,1)],max_batch=0)
        with self.assertRaises(ValueError):c35.sim_disagg_pool([],1,0,1,c32.CostModel(),400)

    def test_disaggregated_work_and_identity(self):
        reqs=[c32.Req(5,0,10,1),c32.Req(9,.1,20,3),c32.Req(17,.2,10,2)]
        for rows in [c35.sim_disagg_pool(reqs,1,2,2,c32.CostModel(),400)[0],
                     c35.sim_colocated(reqs,2,2,c32.CostModel())[0]]:
            self.assertEqual({r.rid for r in rows},{5,9,17})
            self.assertTrue(all(r.generated==r.output for r in rows))
        self.assertTrue(all(r.generated==0 for r in reqs))

    def test_layerwise_transfer_has_an_exposed_tail(self):
        total=c35.transfer_ms(4096,400)
        self.assertAlmostEqual(c35.exposed_ms(4096,400,20000),total/80)
        self.assertAlmostEqual(c35.exposed_ms(4096,400,20000,layers=1),total)

    def test_tp2_layer_budget(self):
        self.assertEqual(c34.KV_PER_TOKEN_LAYER_PER_RANK,2048)
        threshold=c34.DECODE_STEP_S/c34.LAYERS*c34.PCIE_BW/c34.KV_PER_TOKEN_LAYER_PER_RANK
        self.assertAlmostEqual(threshold,7049.560546875)

    def test_draft_kv_capacity_boundary(self):
        target=c37.llama70(1,c37.PEAK_FP8)
        for draft,expected in [(c37.D1B,94),(c37.D8B,40),(c37.EAGLE,111)]:
            batch=c37.max_batch(target,2048,draft)
            self.assertEqual(batch,expected)
            fixed=target.weights_per_gpu()+draft.weights_per_gpu()+c37.RESERVE
            per=2048*(target.kv_per_gpu_token()+draft.kv_per_gpu_token())
            budget=c37.MEM_UTIL*c37.GPU_MEM
            self.assertLessEqual(fixed+batch*per,budget)
            self.assertGreater(fixed+(batch+1)*per,budget)
        self.assertEqual(c37.max_batch(c37.llama70(),2048,c37.D1B),7)

    def test_failed_tail_counts_in_wall_and_cost(self):
        ok=c36.Rec(intended=0,sent=0,first=.1,end=1,ntok=11,ok=True)
        fail=c36.Rec(intended=.2,end=30,ok=False,err='timeout')
        row=c36.summarize([ok,fail],warmup=0,usd_per_gpu_hour=1)
        self.assertAlmostEqual(row['tok_s'],11/30)
        self.assertAlmostEqual(row['usd_per_mtok'],(30/3600)/(11/1e6))
        self.assertEqual(row['failures'],1)

    def test_timeout_has_terminal_timestamp(self):
        async def run():
            async def handler(reader,writer):
                try:
                    await reader.readuntil(b'\r\n\r\n')
                    writer.write(b'HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n')
                    await writer.drain()
                    await reader.read()
                finally:writer.close()
            server=await asyncio.start_server(handler,'127.0.0.1',0)
            try:
                r=c36.Rec(intended=time.perf_counter())
                await c36.one_request(c36.Target(port=server.sockets[0].getsockname()[1]),1,1,r,timeout=.02)
                self.assertFalse(r.ok);self.assertEqual(r.err,'timeout')
                self.assertTrue(math.isfinite(r.end));self.assertGreater(r.end,r.intended)
            finally:server.close();await server.wait_closed()
        asyncio.run(run())

    def test_sse_event_count_is_not_token_count(self):
        async def run():
            async def handler(reader,writer):
                await reader.readuntil(b'\r\n\r\n')
                writer.write(b'HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n'
                             b'data: {"choices":[{"text":"two tokens"}]}\n\ndata: [DONE]\n\n')
                await writer.drain();writer.close()
            server=await asyncio.start_server(handler,'127.0.0.1',0)
            try:
                r=c36.Rec(intended=time.perf_counter())
                await c36.one_request(c36.Target(port=server.sockets[0].getsockname()[1]),1,1,r)
                self.assertEqual(r.nevents,1);self.assertEqual(r.ntok,0);self.assertFalse(r.ok)
            finally:server.close();await server.wait_closed()
        asyncio.run(run())


if __name__=='__main__':unittest.main()
