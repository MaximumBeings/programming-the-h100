#!/usr/bin/env python3
"""Run real streaming requests; compare repeated greedy outputs.

python ch47_vllm_probe.py --model Qwen/Qwen2.5-0.5B-Instruct --output probe.json
The server must already be running on loopback port 8000. No simulated output.
"""
import argparse
import asyncio
import dataclasses
import hashlib
import json
import time
import urllib.request
from ch36_benchmark_teardown import Rec, Target, one_request, summarize


def completion(model, prompt):
    payload = json.dumps(dict(model=model, prompt=prompt, max_tokens=32,
                              temperature=0, seed=42, ignore_eos=True)).encode()
    request = urllib.request.Request('http://127.0.0.1:8000/v1/completions',
                                     data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    return result['choices'][0]['text'], result['usage']


async def run(model):
    target = Target(model=model)
    # Warm-up is excluded from the measured record list.
    warm = Rec(time.perf_counter())
    await one_request(target, 32, 16, warm, timeout=120)
    if not warm.ok:
        raise RuntimeError('warm-up failed: '+warm.err)
    records = []
    for _ in range(8):
        rec = Rec(time.perf_counter())
        await one_request(target, 128, 32, rec, timeout=120)
        if not rec.ok or rec.ntok != 32:
            raise RuntimeError(f'request failed or wrong token count: {rec}')
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    prompt = 'Explain why a KV cache saves recomputation in autoregressive decoding. ' * 12
    a, usage_a = completion(args.model, prompt)
    b, usage_b = completion(args.model, prompt)
    if a != b:
        raise RuntimeError('repeated greedy completion differs')
    records = asyncio.run(run(args.model))
    result = dict(model=args.model, scope='Real single-GPU inference; serial short-request smoke test, not a load sweep',
                  repeated_greedy_equal=True, sample_output=a, usage=[usage_a, usage_b],
                  source_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
                  records=[dataclasses.asdict(r) for r in records],
                  summary=summarize(records,warmup=0))
    with open(args.output,'w') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
