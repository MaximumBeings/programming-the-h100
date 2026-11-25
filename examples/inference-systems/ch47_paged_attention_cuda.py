#!/usr/bin/env python3
"""Educational block-walk attention on CUDA via PyTorch (not a fused kernel).

python ch47_paged_attention_cuda.py
The CPU NumPy reference is Chapter 43's dense_attention. Values are quantized
before the reference so the comparison isolates attention, not input rounding.
"""
import json
import math
import numpy as np
import torch
from ch43_paged_kv_cache import dense_attention


def block_walk(q, keys, values, table, length):
    """q[H,D], pools[P,B,H,D], logical block table; return FP32 [H,D]."""
    if length <= 0:
        raise ValueError('attention requires at least one visible key')
    B = keys.shape[1]
    if len(table) < math.ceil(length / B):
        raise ValueError('block table is too short')
    q = q.float()
    m = torch.full((q.shape[0],), -torch.inf, device=q.device)
    denominator = torch.zeros_like(m)
    numerator = torch.zeros_like(q)
    for logical, physical in enumerate(table[:math.ceil(length / B)]):
        count = min(B, length - logical * B)
        k = keys[physical, :count].float()
        v = values[physical, :count].float()
        scores = torch.einsum('hd,thd->ht', q, k) / math.sqrt(q.shape[-1])
        new_m = torch.maximum(m, scores.max(dim=1).values)
        scale = torch.exp(m - new_m)
        p = torch.exp(scores - new_m[:, None])
        numerator = numerator * scale[:, None] + torch.einsum('ht,thd->hd', p, v)
        denominator = denominator * scale + p.sum(dim=1)
        m = new_m
    return numerator / denominator[:, None]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; no CPU fallback is counted as a pass')
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for length in (1, 3, 17, 65):
            B, H, D = 8, 2, 32
            n = math.ceil(length / B)
            table = list(reversed(range(n)))
            k = torch.randn(n, B, H, D, device='cuda', dtype=dtype)
            v = torch.randn_like(k)
            q = torch.randn(H, D, device='cuda', dtype=dtype)
            # Gather only for independent reference; implementation never gathers.
            kd = k[table].reshape(-1,H,D)[:length].float().cpu().numpy()
            vd = v[table].reshape(-1,H,D)[:length].float().cpu().numpy()
            ref = dense_attention(q.float().cpu().numpy(), kd, vd)
            got = block_walk(q,k,v,table,length).cpu().numpy()
            error = float(np.max(np.abs(got-ref)))
            np.testing.assert_allclose(got,ref,rtol=3e-5,atol=3e-5)
            rows.append(dict(dtype=str(dtype),length=length,max_abs_error=error))
            print(f'PASS {dtype} length={length} max_abs_error={error:.3g}')
    try:
        block_walk(q,k,v,table,0)
    except ValueError:
        print('PASS empty-visible-key input rejected')
    else:
        raise AssertionError('empty input was not rejected')
    print(json.dumps(dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,
                          cases=rows,scope='12 CUDA PyTorch block-walk comparisons; no fused-kernel speed claim'),indent=2))


if __name__ == '__main__':
    main()
