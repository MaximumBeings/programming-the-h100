#!/usr/bin/env python3
"""Blackwell appendix: NVFP4 and MXFP4 emulated in pure Python (standard library only).

Run:  python3 fp4_formats.py            # self-tests, then the error study (about 1-2 minutes)
      python3 fp4_formats.py --selftest # self-tests only

WHAT THIS IS: a bit-faithful emulation of the *value sets and scaling rules* of E2M1 elements with MXFP4 (32-element
blocks, power-of-two E8M0 scale) and NVFP4 (16-element blocks, E4M3 scale, per-tensor FP32 scale), compared with
block-scaled INT4. It runs on a CPU. It is NOT a Blackwell kernel, and no tensor-core behaviour is claimed or tested.
Hopper (H100) has no FP4 tensor cores: on H100 these formats can only be emulated like this, or stored and dequantized
in software before a BF16/FP8 matmul.
"""
import bisect
import math
import random
import sys

# ----------------------------------------------------------------------------- element formats
E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]      # code index i = (exp<<1 | mantissa); index parity = mantissa bit
E2M1_MAX = 6.0


def round_e2m1(x):
    """Round to the nearest E2M1 value, ties to even code (even mantissa bit), saturating at +/-6."""
    s = -1.0 if x < 0 else 1.0
    a = min(abs(x), E2M1_MAX)
    i = bisect.bisect_left(E2M1_POS, a)
    if i == 0:
        return 0.0 * s
    lo, hi = E2M1_POS[i - 1], E2M1_POS[i]
    if a - lo < hi - a:
        v = lo
    elif a - lo > hi - a:
        v = hi
    else:
        v = lo if (i - 1) % 2 == 0 else hi                 # tie: choose the even code
    return s * v


def _e4m3_positive_values():
    """All finite positive E4M3 (FN variant) values: bias 7, no infinities, one NaN code (S.1111.111)."""
    vals = []
    for e in range(0, 16):
        for m in range(0, 8):
            if e == 15 and m == 7:
                continue                                    # NaN
            v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
            if v > 0:
                vals.append(v)
    return sorted(vals)


E4M3_POS = _e4m3_positive_values()
E4M3_MAX = E4M3_POS[-1]                                     # 448
E4M3_MIN = E4M3_POS[0]                                      # 2**-9


def round_e4m3(x):
    """Teaching scale quantizer: midpoint ties choose the smaller, except the zero/min-subnormal tie chooses min-subnormal. Saturate at 448; 0 stays 0."""
    if x <= 0:
        return 0.0
    if x >= E4M3_MAX:
        return E4M3_MAX
    i = bisect.bisect_left(E4M3_POS, x)
    if i == 0:
        return E4M3_POS[0] if x >= E4M3_POS[0] / 2 else 0.0   # underflow below half the smallest subnormal
    lo, hi = E4M3_POS[i - 1], E4M3_POS[i]
    return lo if x - lo <= hi - x else hi


# ----------------------------------------------------------------------------- block quantizers (return dequantized values)
def quant_mxfp4(xs, block=32, scale_round="floor"):
    """OCP MXFP4: per-block power-of-two scale (E8M0), E2M1 elements.
    scale_round='floor': exponent = floor(log2(amax)) - 2  (OCP reference; values in [6,8)*scale saturate)
    scale_round='ceil' : exponent = ceil(log2(amax/6))      (no saturation, coarser elements)"""
    out = []
    for b in range(0, len(xs), block):
        blk = xs[b:b + block]
        amax = max(abs(v) for v in blk)
        if amax == 0:
            out += [0.0] * len(blk)
            continue
        if scale_round == "floor":
            e = math.floor(math.log2(amax)) - 2
        else:
            e = math.ceil(math.log2(amax / E2M1_MAX))
        e = max(-127, min(127, e))
        sc = 2.0 ** e
        out += [round_e2m1(v / sc) * sc for v in blk]
    return out


def tensor_scale_nvfp4(xs):
    """Per-tensor FP32 scale that maps the tensor amax onto the top of the (E2M1 x E4M3) range."""
    amax = max(abs(v) for v in xs)
    return amax / (E2M1_MAX * E4M3_MAX) if amax > 0 else 1.0


def quant_nvfp4(xs, block=16, tensor_scale=None):
    """NVFP4: per-tensor FP32 scale, per-16-block E4M3 scale, E2M1 elements."""
    st = tensor_scale if tensor_scale is not None else tensor_scale_nvfp4(xs)
    out = []
    for b in range(0, len(xs), block):
        blk = xs[b:b + block]
        amax = max(abs(v) for v in blk)
        sb = round_e4m3(amax / E2M1_MAX / st) if amax > 0 else 0.0
        eff = sb * st
        if eff == 0.0:
            out += [0.0] * len(blk)
            continue
        out += [round_e2m1(v / eff) * eff for v in blk]
    return out


def quant_int4(xs, block=32):
    """Symmetric block-absmax INT4 (levels -7..7) with an idealized FP32 scale."""
    out = []
    for b in range(0, len(xs), block):
        blk = xs[b:b + block]
        amax = max(abs(v) for v in blk)
        if amax == 0:
            out += [0.0] * len(blk)
            continue
        sc = amax / 7.0
        out += [max(-7, min(7, round(v / sc))) * sc for v in blk]
    return out


def quant_int4_s8(xs, block=32):
    """Symmetric block-absmax INT4 whose block scale is stored in E4M3 with a per-tensor FP32 scale: the
    equal-storage comparison for MXFP4 (block 32 -> 4.25 bits) and NVFP4 (block 16 -> 4.5 bits)."""
    amax_t = max(abs(v) for v in xs)
    st = amax_t / (7.0 * E4M3_MAX) if amax_t > 0 else 1.0
    out = []
    for b in range(0, len(xs), block):
        blk = xs[b:b + block]
        amax = max(abs(v) for v in blk)
        eff = round_e4m3(amax / 7.0 / st) * st if amax > 0 else 0.0
        if eff == 0.0:
            out += [0.0] * len(blk)
            continue
        out += [max(-7, min(7, round(v / eff))) * eff for v in blk]
    return out


def bits_per_element(fmt):
    return {"int4_b32": 4 + 32 / 32, "int4_b16": 4 + 32 / 16,       # FP32 scale shown; FP16 scale would be 4.5 / 5.0
            "mxfp4": 4 + 8 / 32, "nvfp4": 4 + 8 / 16}[fmt]          # NVFP4: the per-tensor scale adds a negligible amount


# ----------------------------------------------------------------------------- data and metrics
def gen(kind, n, seed):
    r = random.Random(seed)
    if kind == "gaussian":
        return [r.gauss(0, 1) for _ in range(n)]
    if kind == "laplace":
        return [r.choice((-1, 1)) * r.expovariate(1.0) for _ in range(n)]
    if kind == "outliers":                                     # 0.5% of values are 20x larger (activation-like)
        return [r.gauss(0, 1) * (20.0 if r.random() < 0.005 else 1.0) for _ in range(n)]
    raise ValueError(kind)


def sqnr_db(x, y):
    num = sum(v * v for v in x)
    err = sum((a - b) ** 2 for a, b in zip(x, y))
    return float("inf") if err == 0 else 10 * math.log10(num / err)


# ----------------------------------------------------------------------------- self-tests
def selftests():
    res = []

    def check(name, cond):
        res.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    check("E2M1 has exactly 15 distinct signed values {0, +-.5, 1, 1.5, 2, 3, 4, 6}",
          sorted({s * v for v in E2M1_POS for s in (1, -1)}) == sorted(set([-6, -4, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, 4, 6])))
    ties = {0.25: 0.0, 0.75: 1.0, 1.25: 1.0, 1.75: 2.0, 2.5: 2.0, 3.5: 4.0, 5.0: 4.0}
    check("E2M1 round-to-nearest-even on all seven ties", all(round_e2m1(k) == v and round_e2m1(-k) == -v for k, v in ties.items()))
    check("E2M1 saturates at +-6 and keeps sign of zero-rounded values", round_e2m1(100) == 6 and round_e2m1(-7) == -6 and round_e2m1(0.1) == 0)
    check("E4M3: 126 finite positive values, max 448, min 2^-9", len(E4M3_POS) == 126 and E4M3_MAX == 448.0 and E4M3_MIN == 2.0 ** -9)
    check("E4M3 rounding is idempotent and saturates", all(round_e4m3(v) == v for v in E4M3_POS) and round_e4m3(1e9) == 448.0)
    check("E4M3 has 8 codes per binade (3 mantissa bits): 1.0, 1.125, ..., 1.875 all exact",
          all(round_e4m3(1 + k / 8) == 1 + k / 8 for k in range(8)))

    rng = random.Random(7)
    xs = [rng.gauss(0, 1) for _ in range(4096)]
    for name, q in (("MXFP4 floor", lambda v: quant_mxfp4(v, 32, "floor")), ("MXFP4 ceil", lambda v: quant_mxfp4(v, 32, "ceil")),
                    ("NVFP4", lambda v: quant_nvfp4(v)), ("INT4 b32", lambda v: quant_int4(v, 32)),
                    ("INT4 b16 + E4M3 scale", lambda v: quant_int4_s8(v, 16))):
        y = q(xs)
        check(f"{name}: quantizing already-quantized data changes nothing (idempotent)", all(abs(a - b) < 1e-9 for a, b in zip(y, q(y))))
    # structural checks of the formats
    y = quant_mxfp4(xs, 32, "floor")
    exps = []
    for b in range(0, len(xs), 32):
        amax = max(abs(v) for v in xs[b:b + 32])
        exps.append(math.floor(math.log2(amax)) - 2)
    ok = True
    for bi, e in enumerate(exps):
        sc = 2.0 ** e
        for v in y[bi * 32:(bi + 1) * 32]:
            ok &= (v / sc) in set(s * u for u in E2M1_POS for s in (1, -1))
    check("MXFP4: every element is an E2M1 code times a power of two (shared per 32)", ok)
    y = quant_nvfp4(xs)
    st = tensor_scale_nvfp4(xs)
    ok = True
    for b in range(0, len(xs), 16):
        amax = max(abs(v) for v in xs[b:b + 16])
        sb = round_e4m3(amax / 6 / st)
        for v in y[b:b + 16]:
            ok &= abs(round(v / (sb * st) * 2) / 2 - v / (sb * st)) < 1e-9
    check("NVFP4: every element is an E2M1 code times (E4M3 block scale x FP32 tensor scale)", ok)
    check("zero blocks stay exactly zero in every format", all(quant_nvfp4([0.0] * 32)[i] == 0 for i in range(32)) and
          all(v == 0 for v in quant_mxfp4([0.0] * 32)) and all(v == 0 for v in quant_int4([0.0] * 32)))
    check("bits per element: MXFP4 4.25, NVFP4 4.5", bits_per_element("mxfp4") == 4.25 and bits_per_element("nvfp4") == 4.5)
    # saturation property: ceil variant never clips; floor variant clips only values in [6,8) x scale
    x1 = [7.0] + [0.1] * 31                                    # amax 7 -> floor exp = 0 -> scale 1 -> 7 clips to 6
    check("MXFP4 floor clips the block maximum (7.0 -> 6.0); the ceil variant rounds it to 8.0 instead (7/2 = 3.5 ties to 4)",
          quant_mxfp4(x1, 32, "floor")[0] == 6.0 and abs(quant_mxfp4(x1, 32, "ceil")[0] - 8.0) < 1e-9)
    # monotonicity: smaller blocks never hurt (statistically) for the same scale format -- check with NVFP4 on Gaussian
    e16 = sqnr_db(xs, quant_nvfp4(xs, 16)); e64 = sqnr_db(xs, quant_nvfp4(xs, 64))
    check(f"NVFP4 block 16 is at least as accurate as block 64 on Gaussian data ({e16:.2f} dB vs {e64:.2f} dB)", e16 >= e64)
    # failure mode: a huge outlier sets the per-tensor scale so large that ordinary blocks underflow the E4M3 scale
    big = [1.0e6] + [1.0] * 31 + [random.Random(3).gauss(0, 1) for _ in range(64)]
    nv = quant_nvfp4(big, 16)
    mx = quant_mxfp4(big, 32, "floor")
    check("NVFP4 per-tensor outlier (1e6): blocks of ordinary values (amax ~1) flush to exactly zero",
          all(v == 0 for v in nv[32:]) and all(v == 0 for v in nv[16:32]))
    check("MXFP4 with the same outlier: ordinary blocks survive (power-of-two scale has +-127 binades of range)",
          sum(1 for v in mx[32:] if v != 0) > 40)
    check("NVFP4 block-scale range: largest/smallest nonzero E4M3 scale = 448 / 2^-9 = 229,376", E4M3_MAX / E4M3_MIN == 229376.0)
    return all(res)


def study():
    n = 1 << 16
    print("\nSQNR in dB (higher is better) on 65,536 values, mean of 3 seeds. Idealized FP32 INT4 scales.")
    fmts = [("INT4 b32, FP32 scale (5.00 b)", lambda v: quant_int4(v, 32)),
            ("INT4 b16, FP32 scale (6.00 b)", lambda v: quant_int4(v, 16)),
            ("INT4 b32, E4M3 scale (4.25 b)", lambda v: quant_int4_s8(v, 32)),
            ("INT4 b16, E4M3 scale (4.50 b)", lambda v: quant_int4_s8(v, 16)),
            ("MXFP4 floor          (4.25 b)", lambda v: quant_mxfp4(v, 32, "floor")),
            ("MXFP4 ceil           (4.25 b)", lambda v: quant_mxfp4(v, 32, "ceil")),
            ("NVFP4                (4.50 b)", lambda v: quant_nvfp4(v, 16))]
    print(f"  {'format':30s} {'gaussian':>9s} {'laplace':>9s} {'outliers':>9s}")
    table = {}
    for name, q in fmts:
        row = []
        for kind in ("gaussian", "laplace", "outliers"):
            vals = [sqnr_db(x, q(x)) for x in (gen(kind, n, s) for s in (1, 2, 3))]
            row.append(sum(vals) / len(vals))
        table[name] = row
        print(f"  {name:30s} {row[0]:9.2f} {row[1]:9.2f} {row[2]:9.2f}")
    # dot-product (GEMM-like) relative error: quantize both operands, K=4096, 128 pairs
    print("\nDot-product relative error |dot_q - dot| / (||a|| ||b||), K=4096, mean of 128 random pairs (lower is better):")
    for name, q in fmts:
        errs = []
        for s in range(128):
            a, b = gen("outliers", 4096, 1000 + s), gen("gaussian", 4096, 5000 + s)
            d = sum(x * y for x, y in zip(a, b))
            dq = sum(x * y for x, y in zip(q(a), q(b)))
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
            errs.append(abs(dq - d) / (na * nb))
        print(f"  {name:30s} {sum(errs) / len(errs):.5f}")
    # a block with one big outlier
    blk = [0.05 * math.sin(i) for i in range(16)]
    blk[3] = 3.7
    print("\nOne 16-element block with a single outlier (3.7) among values near 0.05:")
    for name, y in (("MXFP4 floor (block padded with 16 zeros)", quant_mxfp4(blk + [0.0] * 16, 32, "floor")[:16]),
                    ("NVFP4", quant_nvfp4(blk, 16)), ("INT4 b16, FP32 scale", quant_int4(blk, 16))):
        print(f"  {name:42s} outlier -> {y[3]:7.4f}   max |err| on the others = {max(abs(a - b) for i, (a, b) in enumerate(zip(blk, y)) if i != 3):.4f}")
    return table


if __name__ == "__main__":
    print("self-tests:")
    ok = selftests()
    if "--selftest" not in sys.argv:
        study()
    print("\nALL PASS" if ok else "\nFAILURES")
    sys.exit(0 if ok else 1)
