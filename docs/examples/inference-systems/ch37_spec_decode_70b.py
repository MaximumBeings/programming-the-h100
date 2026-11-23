#!/usr/bin/env python3
"""Chapter 37: speculative decoding at 70B -- cost model, Monte Carlo, property tests.
Pure standard library. Run:  python3 ch37_spec_decode_70b.py
Nothing here was run on an H100; hardware constants are published peak figures
and `mfu` / `overhead` are ASSUMPTIONS you must replace with measurements."""
import math, random, statistics

# ---------- hardware / model description ----------
HBM_BW = 3.35e12          # H100 SXM bytes/s (peak)
PEAK_BF16 = 989e12        # dense FLOP/s per GPU (peak)

class Model:
    def __init__(s, name, params, bytes_per_param, gpus):
        s.name, s.params, s.bpp, s.gpus = name, params, bytes_per_param, gpus
    def weight_bytes_per_gpu(s): return s.params * s.bpp / s.gpus

def step_time(m, tokens, mfu=0.5, overhead=1.0e-3, comm=0.0):
    """One forward over `tokens` total tokens (batch x tokens-per-seq).
    Roofline: max(memory time, compute time) + fixed overhead + comm."""
    mem = m.weight_bytes_per_gpu() / HBM_BW
    comp = 2.0 * m.params * tokens / (m.gpus * PEAK_BF16 * mfu)
    return max(mem, comp) + overhead + comm

def expected_tokens(alpha, K):
    """E[tokens emitted per verify pass] for i.i.d. per-token acceptance alpha."""
    if alpha >= 1.0: return K + 1.0
    return (1.0 - alpha ** (K + 1)) / (1.0 - alpha)

def cycle_time(target, draft, K, batch, mfu=0.5, comm=0.0):
    td = K * step_time(draft, batch, mfu)                 # K sequential draft steps
    tv = step_time(target, batch * (K + 1), mfu, comm=comm)  # one verify, K+1 tokens/seq
    return td + tv

def speedup(target, draft, alpha, K, batch, mfu=0.5, comm=0.0):
    base = step_time(target, batch, mfu, comm=comm)
    return expected_tokens(alpha, K) * base / cycle_time(target, draft, K, batch, mfu, comm)

def best_K(target, draft, alpha, batch, Kmax=12, **kw):
    best = max(range(1, Kmax + 1), key=lambda k: speedup(target, draft, alpha, k, batch, **kw))
    return best, speedup(target, draft, alpha, best, batch, **kw)

# ---------- Monte Carlo of the accept/reject process ----------
def simulate_tokens_per_pass(alpha, K, n=200000, seed=0):
    rng = random.Random(seed); tot = 0
    for _ in range(n):
        acc = 0
        while acc < K and rng.random() < alpha: acc += 1
        tot += acc + 1          # accepted prefix + 1 corrected/bonus token
    return tot / n

# ---------- acceptance from distributions (the real alpha) ----------
def acceptance(p, q):
    """Per-position acceptance prob of rejection sampling: sum_x min(p(x), q(x))."""
    return sum(min(a, b) for a, b in zip(p, q))

def softmax(z, T):
    m = max(z); e = [math.exp((x - m) / T) for x in z]; s = sum(e); return [x / s for x in e]

def alpha_vs_temperature(seed=1, V=50, noise=0.8, temps=(0.2, 0.5, 1.0, 1.5)):
    """Draft logits = target logits + noise. The direction of the trend is a property of this synthetic noise model, NOT of real models."""
    rng = random.Random(seed); out = {}
    for T in temps:
        vals = []
        for _ in range(400):
            zt = [rng.gauss(0, 3) for _ in range(V)]
            zd = [x + rng.gauss(0, noise) for x in zt]
            vals.append(acceptance(softmax(zt, T), softmax(zd, T)))
        out[T] = statistics.mean(vals)
    return out

# ---------- self tests ----------
def selftests():
    t = Model("70B-bf16-tp2", 70e9, 2, 2)
    d = Model("8B-bf16-tp1", 8e9, 2, 1)
    r = []
    def chk(name, ok, info=""):
        r.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")
    # 1 decode floor
    floor = t.weight_bytes_per_gpu() / HBM_BW * 1e3
    chk("70B TP2 decode floor ~20.9 ms", abs(floor - 20.9) < 0.1, f"({floor:.2f} ms)")
    # 2 formula vs Monte Carlo
    errs = [abs(simulate_tokens_per_pass(a, K) - expected_tokens(a, K))
            for a in (0.3, 0.6, 0.8, 0.9) for K in (1, 3, 5, 8)]
    chk("E[tokens] formula matches Monte Carlo", max(errs) < 0.02, f"(max err {max(errs):.4f})")
    # 3 alpha=0 -> exactly 1 token ; alpha=1 -> K+1
    chk("limits: alpha=0 gives 1, alpha=1 gives K+1",
        expected_tokens(0, 5) == 1.0 and expected_tokens(1, 5) == 6.0)
    # 4 speedup monotone in alpha
    s = [speedup(t, d, a, 4, 1) for a in (0.2, 0.4, 0.6, 0.8, 0.95)]
    chk("speedup increases with alpha", all(x < y for x, y in zip(s, s[1:])))
    # 5 optimal K nondecreasing in alpha (batch 1)
    ks = [best_K(t, d, a, 1)[0] for a in (0.3, 0.5, 0.7, 0.9)]
    chk("best K nondecreasing in alpha", all(x <= y for x, y in zip(ks, ks[1:])), f"({ks})")
    # 6 speedup nonincreasing in batch once verify turns compute-bound
    sb = [speedup(t, d, 0.8, 4, b) for b in (1, 8, 32, 128, 512)]
    chk("speedup nonincreasing in batch", all(x >= y - 1e-9 for x, y in zip(sb, sb[1:])), f"({[round(x,2) for x in sb]})")
    # 7 acceptance identity: alpha(p,p)=1, disjoint support=0
    p = [0.5, 0.5, 0.0]; q = [0.0, 0.0, 1.0]
    chk("acceptance(p,p)=1, disjoint=0", abs(acceptance(p, p) - 1) < 1e-12 and acceptance(p, q) == 0)
    # 8 a bad draft can be slower than no speculation
    slow = Model("slow-draft", 70e9, 2, 2)  # draft as big as target
    chk("draft as large as target gives speedup < 1", speedup(t, slow, 0.7, 4, 1) < 1.0)
    return all(r)

def tables():
    t = Model("70B", 70e9, 2, 2); d8 = Model("8B", 8e9, 2, 1); d1 = Model("1B", 1e9, 2, 1)
    print("\nBase step (70B BF16, TP2):")
    for b in (1, 8, 32, 64, 128, 256, 512):
        print(f"  batch {b:4d}: {step_time(t, b)*1e3:6.1f} ms")
    for d in (d8, d1):
        print(f"\nSpeedup, draft={d.name}, K=best in 1..12, mfu=0.5, overhead=1ms (ASSUMED)")
        print("  alpha  " + "".join(f"B={b:<10d}" for b in (1, 8, 32, 128, 512)))
        for a in (0.5, 0.7, 0.8, 0.9):
            row = ""
            for b in (1, 8, 32, 128, 512):
                k, s = best_K(t, d, a, b)
                row += f"{s:4.2f}x(K={k:<2d}) "
            print(f"  {a:.1f}    {row}")
    print("\nalpha vs temperature (draft = target logits + N(0,0.8) noise, synthetic):")
    for T, a in alpha_vs_temperature().items(): print(f"  T={T:<4}: alpha={a:.3f}")

if __name__ == "__main__":
    print("self-tests:"); ok = selftests(); tables()
    print("\nALL PASS" if ok else "\nFAILURES"); raise SystemExit(0 if ok else 1)
