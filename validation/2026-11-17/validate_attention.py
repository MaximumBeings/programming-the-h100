#!/usr/bin/env python3
"""Numerical reference for Chapter 12's online softmax, not an FA3 benchmark."""
import datetime
import math
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


@torch.no_grad()
def streaming_attention(q, k, v, causal, block=16):
    rows, width = q.shape
    maximum = torch.full((rows, 1), -torch.inf, device=q.device)
    denominator = torch.zeros((rows, 1), device=q.device)
    numerator = torch.zeros_like(q)
    for start in range(0, k.shape[0], block):
        scores = q @ k[start:start + block].T / math.sqrt(width)
        if causal:
            row = torch.arange(rows, device=q.device)[:, None]
            col = torch.arange(start, min(start + block, k.shape[0]), device=q.device)[None, :]
            scores.masked_fill_(col > row, -torch.inf)
        updated = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
        correction = torch.exp(maximum - updated)
        probabilities = torch.exp(scores - updated)
        numerator = numerator * correction + probabilities @ v[start:start + block]
        denominator = denominator * correction + probabilities.sum(dim=-1, keepdim=True)
        maximum = updated
    return numerator / denominator


def main():
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    print('Execution UTC:', datetime.datetime.now(datetime.timezone.utc).isoformat())
    print('GPU:', torch.cuda.get_device_name())
    print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
    print('Reference algorithm only: no FA3 kernel or throughput claim.')
    for length in (17, 33, 65):
        for width in (32, 64):
            q, k, v = [torch.randn(length, width, device='cuda') for _ in range(3)]
            for causal in (False, True):
                got = streaming_attention(q, k, v, causal)
                with sdpa_kernel(SDPBackend.MATH):
                    expected = torch.nn.functional.scaled_dot_product_attention(
                        q[None, None], k[None, None], v[None, None], is_causal=causal)[0, 0]
                torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-5)
                error = (got - expected).abs().max().item()
                print(f'N={length} d={width} causal={causal}: PASS max_abs_error={error:.9f}')
    print('PASS: 12 cases, including causal masks and partial final tiles.')


if __name__ == '__main__':
    main()
