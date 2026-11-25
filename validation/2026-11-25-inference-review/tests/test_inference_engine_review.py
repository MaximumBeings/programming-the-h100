#!/usr/bin/env python3
"""Adversarial CPU boundary tests for the educational engine and scheduler."""
import math
import sys
from pathlib import Path
import unittest
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'docs/examples/inference-systems'))
from ch43_paged_kv_cache import PagedKVCache, paged_attention
from ch44_continuous_batching import Req, simulate_continuous
from ch45_radix_prefix_cache import RadixCache, BlockHashCache, CacheFull
from ch46_mini_engine import TinyLM, Engine


class EngineReviewTests(unittest.TestCase):
    def test_empty_attention_is_not_a_nan_result(self):
        cache=PagedKVCache(2,4,2,8);cache.add_seq('empty')
        with self.assertRaisesRegex(ValueError,'visible key'):
            paged_attention(np.zeros((2,8)),cache,'empty')

    def test_request_lengths_and_arrivals_rejected(self):
        for prompt,out,arrival in [(0,2,0),(2,0,0),(2,-1,0),(2,2,math.nan),(2,2,math.inf),(2,2,-1)]:
            with self.subTest(prompt=prompt,out=out,arrival=arrival):
                with self.assertRaises(ValueError):Req('s',arrival,prompt,out)

    def test_scheduler_configuration_rejected(self):
        for kwargs in [dict(n_blocks=0),dict(B=0),dict(max_seqs=0),dict(max_batched_tokens=0),
                       dict(max_batched_tokens=-1),dict(max_batched_tokens=1.5),dict(watermark=1),
                       dict(watermark=math.nan),dict(cp=-1)]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):simulate_continuous([],**kwargs)

    def test_engine_rejects_bad_requests_without_allocating(self):
        model=TinyLM(max_pos=8)
        for prompt,out,arrival in [([],2,0),([1],0,0),([-1],2,0),([model.V],2,0),
                                   ([1.5],2,0),([1],9,0),([1],1,math.nan)]:
            with self.subTest(prompt=prompt,out=out,arrival=arrival):
                eng=Engine(model,n_blocks=4,B=2,prefix_cache=True)
                with self.assertRaises(ValueError):eng.run([('s',arrival,prompt,out)])
                self.assertEqual(eng.alloc.num_free(),4)
                self.assertEqual(eng.cache.nodes,{})

    def test_engine_checks_entire_batch_before_admission(self):
        eng=Engine(TinyLM(),n_blocks=4,B=2,prefix_cache=True)
        with self.assertRaises(ValueError):eng.run([('valid',0,[1,2],2),('invalid',0,[-1],2)])
        self.assertEqual(eng.alloc.num_free(),4)
        self.assertEqual(eng.cache.nodes,{})

    def test_engine_positional_boundary_is_exact(self):
        model=TinyLM(max_pos=4)
        # Last emitted token is not fed back: 2 prompt + 3 outputs need 4 positions.
        eng=Engine(model,n_blocks=8,B=2,debug=True)
        self.assertTrue(np.isfinite(model.forward_full([1,2])).all())
        result=eng.run([('s',0,[1,2],3)])
        self.assertEqual(result[0].out,model.ref_generate([1,2],3))
        self.assertEqual(eng.alloc.num_free(),8)

    def test_engine_configuration(self):
        for kwargs in [dict(B=0),dict(max_seqs=0),dict(budget=0),dict(budget=1.5),dict(watermark=1)]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):Engine(TinyLM(),**kwargs)
        for kwargs in [dict(d=3,H=1),dict(d=32,H=3),dict(V=0)]:
            with self.assertRaises(ValueError):TinyLM(**kwargs)

    def test_cachefull_match_is_not_a_pinned_handle(self):
        cache=RadixCache(3);cache.insert((1,2,3))
        with self.assertRaises(CacheFull) as raised:cache.acquire((1,2,3,4))
        self.assertEqual(raised.exception.matched,3)
        # Failure returns a length, not ownership: another operation can evict it.
        self.assertEqual(cache.evict(3),3)
        self.assertEqual(cache.match_length((1,2,3)),0)
        cache.check_invariants()

    def test_matmul_outputs_match_independent_non_blas_forward(self):
        model=TinyLM(max_pos=64)
        def mm(a,b):
            return np.einsum('ij,jk->ik',a,b,optimize=False)
        def rms(x):
            return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)
        def reference(tokens):
            x=model.E[tokens]+model.P[:len(tokens)]
            mask=np.tril(np.ones((len(tokens),len(tokens)),bool))
            for weights in model.layers:
                h=rms(x)
                q=mm(h,weights['Wq']).reshape(-1,model.H,model.D)
                k=mm(h,weights['Wk']).reshape(-1,model.H,model.D)
                v=mm(h,weights['Wv']).reshape(-1,model.H,model.D)
                scores=np.einsum('qhd,khd->hqk',q,k)/math.sqrt(model.D)
                scores=np.where(mask[None],scores,-np.inf)
                scores-=scores.max(-1,keepdims=True)
                prob=np.exp(scores);prob/=prob.sum(-1,keepdims=True)
                attn=np.einsum('hqk,khd->qhd',prob,v)
                x=x+mm(attn.reshape(len(tokens),-1),weights['Wo'])
                h=rms(x)
                x=x+mm(np.maximum(mm(h,weights['W1']),0),weights['W2'])
            return mm(rms(x),model.E.T)
        for length in (1,2,3,4,7,16,31):
            tokens=list(range(length))
            # Inspect warnings only after independently validating finite values and
            # every output element. Never infer correctness from argmax agreement.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',RuntimeWarning)
                got=model.forward_full(tokens)
            expected=reference(tokens)
            self.assertTrue(np.isfinite(got).all())
            self.assertTrue(np.isfinite(expected).all())
            np.testing.assert_allclose(got,expected,rtol=3e-13,atol=3e-13)
            print(f'BLAS diagnostic: length={length}, warnings={len(caught)}, finite=True, '
                  f'max_abs_error={np.max(np.abs(got-expected)):.3g}')
            for message in sorted({str(w.message) for w in caught}):
                print('  captured warning:',message)

    def test_cache_configuration(self):
        for kwargs in [dict(capacity=-1),dict(capacity=2,policy='random')]:
            with self.assertRaises(ValueError):RadixCache(**kwargs)
        for kwargs in [dict(capacity_tokens=4,B=0),dict(capacity_tokens=4,B=8),
                       dict(capacity_tokens=8,B=8,hash_bits=0)]:
            with self.assertRaises(ValueError):BlockHashCache(**kwargs)


if __name__=='__main__':unittest.main(verbosity=2)
