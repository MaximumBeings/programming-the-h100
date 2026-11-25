#!/usr/bin/env python3
"""Adversarial regressions: numerical masks, cache state, causal events and SLO accounting."""
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'docs/examples/inference-systems'))
import ch38_semantic_cache as c38
import ch39_complexity_router as c39
import ch40_gpu_autoscaler as c40
import ch41_edge_fallback as c41
import ch42_gateway_capstone as c42


def rejects(f):
    try:
        f()
    except ValueError:
        return
    raise AssertionError('invalid input was accepted')


def main():
    # True zero probabilities are common after vocabulary masks.
    assert c39.token_entropy_nats(np.array([[0.0, -np.inf]]))[0] == 0.0
    np.testing.assert_allclose(c39.token_entropy_nats(np.array([[0.0, 0.0, -np.inf]])), [math.log(2)])
    rejects(lambda:c39.token_entropy_nats(np.array([[-np.inf, -np.inf]])))
    rejects(lambda:c39.token_entropy_nats(np.array([[0.0, np.nan]])))
    print('PASS masked entropy uses 0 log 0 and rejects undefined distributions')

    cache=c38.SemanticCache(2, 0.9, max_entries=1)
    rejects(lambda:cache.insert('k',np.array([0.0,0.0]),'bad'))
    rejects(lambda:cache.insert('k',np.array([np.nan,1.0]),'bad'))
    assert cache.size()==0 and not cache.parts
    rejects(lambda:c38.SemanticCache(2,0.9,max_entries=0))
    assert c38.breakeven_bad_cost(0.0,0.0,1.0,0.1)==-math.inf
    print('PASS invalid cache input leaves state unchanged; zero-wrong unprofitable cache is not an infinite win')

    # At a drain horizon, censored requests have lower-bound waits, not successes.
    result=c40.simulate(np.array([9.8]),np.array([1.0]),10.0,static=0,dt=0.1)
    assert not result['started'][0]
    assert c40.summarize(result,5.0,4.0)['attain']==0.0
    assert c40.summarize(result,5.0,4.0,window=(9,10),arrivals=np.array([9.8]))['attain']==0.0
    print('PASS unstarted censored requests stay in SLO denominator and cannot count as successes')

    # Overlapping requests must see a closed breaker until failures really arrive.
    base=c41.CircuitBreaker
    class ObservedBreaker(base):
        times=[]
        def __init__(self):
            super().__init__(threshold=1,open_for=10.0)
        def record(self,gen,ok,now):
            self.times.append(now)
            return super().record(gen,ok,now)
    reqs=[c41.Req(61.0,100,100),c41.Req(61.1,100,100),c41.Req(61.2,100,100)]
    with patch.object(c41,'CircuitBreaker',ObservedBreaker),patch.object(c41,'make_requests',return_value=reqs):
        got=c41.run_client(c41.Policy('overlap',breaker=True),'hang',0)
    assert got['remote_per_req']==1.0
    assert ObservedBreaker.times==sorted(ObservedBreaker.times)
    print('PASS overlapping breaker results are delivered chronologically')

    # Force cold start/queueing, then verify each reusable answer really existed.
    g=c42.gateway(c42.Cfg('causal',cache=True,router=True,autoscale=True),2)
    assert g['cache_hit_events']
    for producer,inserted,hit in g['cache_hit_events']:
        assert g['recs'][producer].ok
        assert inserted==g['recs'][producer].end and inserted<=hit
    for rid,_start,finished in g['prefix_log']:
        if g['recs'][rid].ok:
            assert g['recs'][rid].first>=finished
    print('PASS cached answers follow actual producing completions and routed responses follow prefix completion')

    # A direct-small request cannot publish its first token before completing
    # the same K-token prefix used to decide whether to escalate.
    model_data=dict(H=np.zeros(4),cs=np.ones(4,bool),cl=np.ones(4,bool))
    with patch.object(c42,'make_arrivals',return_value=np.array([0.05])), \
         patch.object(c42,'make_data',return_value=model_data), \
         patch.object(c42,'calibrate_tau',return_value=1.0), \
         patch.object(c42,'lognormal_service',return_value=np.array([2.0])), \
         patch.object(c42,'FAMILIES',1):
        one=c42.gateway(c42.Cfg('direct-small',router=True),0)
    assert one['kind']==['small'] and one['recs'][0].ok
    assert abs(one['recs'][0].first-(0.1+0.2*2.0))<1e-12
    print('PASS direct-small first content waits for its routing prefix (0.5s, not 0.25s)')


if __name__=='__main__':
    main()
