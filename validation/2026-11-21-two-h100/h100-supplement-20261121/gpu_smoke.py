"""Compiled NVIDIA smoke tests. GPU execution is pending; never use interpreter mode."""
import datetime
import json
import math
import os
import platform
import sys

if not __debug__:
    raise SystemExit("BLOCKED: run without -O or PYTHONOPTIMIZE; numerical assertions are required")

if os.environ.get("TRITON_INTERPRET") == "1":
    raise SystemExit("BLOCKED: unset TRITON_INTERPRET for a real GPU test")

import torch
import triton
import triton.language as tl


@triton.jit
def add_strided(X, Y, Z, N, SX, SY, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i * SX, i < N, other=0.0)
    y = tl.load(Y + i * SY, i < N, other=0.0)
    tl.store(Z + i, x + y, i < N)


@triton.jit
def matmul(A, B, C, M, N, K, TILE: tl.constexpr):
    m = tl.program_id(0) * TILE + tl.arange(0, TILE)
    n = tl.program_id(1) * TILE + tl.arange(0, TILE)
    kk = tl.arange(0, TILE)
    acc = tl.zeros((TILE, TILE), tl.float32)
    for start in range(0, K, TILE):
        k = start + kk
        a = tl.load(A + m[:, None] * K + k[None, :],
                    (m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(B + k[:, None] * N + n[None, :],
                    (k[:, None] < K) & (n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(C + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def attention(Q, K, V, O, M, N, D, SCALE: tl.constexpr,
              CAUSAL: tl.constexpr, BD: tl.constexpr, TILE: tl.constexpr):
    rows = tl.program_id(0) * TILE + tl.arange(0, TILE)
    ds = tl.arange(0, BD)
    qs = tl.load(Q + rows[:, None] * D + ds[None, :],
                 (rows[:, None] < M) & (ds[None, :] < D), other=0.0)
    maximum = tl.full((TILE,), float("-inf"), tl.float32)
    denominator = tl.zeros((TILE,), tl.float32)
    numerator = tl.zeros((TILE, BD), tl.float32)
    end = N
    if CAUSAL:
        end = tl.minimum(N, (tl.program_id(0) + 1) * TILE)
    for start in range(0, end, TILE):
        cols = start + tl.arange(0, TILE)
        valid_k = (cols[:, None] < N) & (ds[None, :] < D)
        ks = tl.load(K + cols[:, None] * D + ds[None, :], valid_k, other=0.0)
        vs = tl.load(V + cols[:, None] * D + ds[None, :], valid_k, other=0.0)
        scores = tl.dot(qs, tl.trans(ks), input_precision="ieee") * SCALE
        valid = cols[None, :] < N
        if CAUSAL:
            valid = valid & (rows[:, None] >= cols[None, :])
        scores = tl.where(valid, scores, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(scores, axis=1))
        rescale = tl.exp(maximum - next_max)
        weights = tl.exp(scores - next_max[:, None])
        denominator = rescale * denominator + tl.sum(weights, axis=1)
        numerator = numerator * rescale[:, None] + tl.dot(
            weights, vs, input_precision="ieee")
        maximum = next_max
    output = numerator / denominator[:, None]
    tl.store(O + rows[:, None] * D + ds[None, :], output,
             (rows[:, None] < M) & (ds[None, :] < D))


def main():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise SystemExit("BLOCKED: this smoke test requires a CUDA-visible NVIDIA GPU")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(json.dumps({
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "torch": torch.__version__,
        "triton": triton.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "scope": "compiled GPU correctness; no performance measurement",
    }, indent=2))
    count = 0
    for n in (1, 137, 4099):
        x_cpu = torch.arange(n * 2, dtype=torch.float32)[::2]
        y_cpu = torch.arange(n * 3, dtype=torch.float32)[::3]
        x = torch.arange(n * 2, device="cuda", dtype=torch.float32)[::2]
        y = torch.arange(n * 3, device="cuda", dtype=torch.float32)[::3]
        guarded = torch.full((n + 2,), -12345.0, device="cuda")
        out = guarded[1:-1]
        add_strided[(triton.cdiv(n, 128),)](
            x, y, out, n, x.stride(0), y.stride(0), BLOCK=128)
        assert torch.equal(out.cpu(), x_cpu + y_cpu)
        assert guarded[0].item() == -12345.0 and guarded[-1].item() == -12345.0
        print(f"PASS strided-add N={n}; complete output and both guards")
        count += 1
    for m, n, k in ((17, 23, 19), (16, 16, 16), (1, 7, 33)):
        a = ((torch.arange(m * k).reshape(m, k) % 17) - 8).float() / 16
        b = ((torch.arange(k * n).reshape(k, n) % 13) - 6).float() / 16
        reference = (a.double() @ b.double()).float()
        out = torch.empty((m, n), device="cuda")
        matmul[(triton.cdiv(m, 16), triton.cdiv(n, 16))](
            a.cuda(), b.cuda(), out, m, n, k, TILE=16)
        assert torch.equal(out.cpu(), reference), "binary-fraction GEMM mismatch"
        print(f"PASS ragged-gemm M={m} N={n} K={k}; complete CPU FP64 oracle")
        count += 1
    for m, n, d in ((17, 23, 8), (20, 20, 8), (3, 19, 16)):
        q = ((torch.arange(m * d).reshape(m, d) % 11) - 5).float() / 8
        k = ((torch.arange(n * d).reshape(n, d) % 13) - 6).float() / 8
        v = ((torch.arange(n * d).reshape(n, d) % 17) - 8).float() / 8
        for causal in (False, True):
            scores = q.double() @ k.double().T / math.sqrt(d)
            if causal:
                allowed = torch.arange(n)[None, :] <= torch.arange(m)[:, None]
                scores = scores.masked_fill(~allowed, -math.inf)
            reference = (scores.softmax(dim=-1) @ v.double()).float()
            out = torch.empty((m, d), device="cuda")
            attention[(triton.cdiv(m, 16),)](
                q.cuda(), k.cuda(), v.cuda(), out, m, n, d,
                SCALE=1 / math.sqrt(d), CAUSAL=causal,
                BD=max(16, triton.next_power_of_2(d)), TILE=16)
            actual = out.cpu()
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
            print(f"PASS attention M={m} N={n} D={d} causal={causal}; CPU FP64 oracle")
            count += 1
    torch.cuda.synchronize()
    assert count == 12
    print("GPU SMOKE PASS: 12 compiled GPU cases; not the complete chapter suite")


if __name__ == "__main__":
    main()
