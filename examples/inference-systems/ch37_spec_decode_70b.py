#!/usr/bin/env python3
"""Chapter 37: speculative decoding at 70B -- roofline cost model with KV cache, memory feasibility, tests.
Pure standard library. Run:  python3 ch37_spec_decode_70b.py
Nothing here was run on an H100. Hardware constants are published peak figures; `mfu`, the per-all-reduce latency,
the fixed per-step overhead and every acceptance rate alpha are ASSUMPTIONS to be replaced by measurements."""
import math, random, statistics

# ---------- hardware ----------
HBM_BW = 3.35e12            # bytes/s, H100 SXM peak
PEAK_BF16 = 989e12          # dense FLOP/s per GPU
PEAK_FP8 = 1979e12
GPU_MEM = 80 * 2**30        # 80 GiB = 85.9e9 bytes
MEM_UTIL = 0.90             # fraction of GPU memory the engine may use (vLLM-style gpu_memory_utilization)
RESERVE = 2.0e9             # activations, CUDA graphs, workspace per GPU (assumed)
AR_LATENCY = 12e-6          # one small NVLink all-reduce (ASSUMED, measure with nccl-tests)

class Model:
    def __init__(s, name, params, bpp, gpus, layers, kv_heads, head_dim, peak=PEAK_BF16, kv_bpp=2):
        s.name, s.params, s.bpp, s.gpus, s.layers = name, params, bpp, gpus, layers
        s.peak = peak
        s.kv_bytes_per_token = 2 * layers * kv_heads * head_dim * kv_bpp      # K and V, all layers, whole model
    def weights_per_gpu(s): return s.params * s.bpp / s.gpus
    def kv_per_gpu_token(s): return s.kv_bytes_per_token / s.gpus             # KV heads are sharded across TP ranks
    def comm_s(s): return 2 * s.layers * AR_LATENCY if s.gpus > 1 else 0.0   # 2 all-reduces per layer

def llama70(bpp=2, peak=PEAK_BF16): return Model("70B", 70e9, bpp, 2, 80, 8, 128, peak)
def draft(name, params, layers, kv_heads, head_dim): return Model(name, params, 2, 1, layers, kv_heads, head_dim)
D1B = draft("1B", 1.2e9, 16, 8, 64)        # Llama-3.2-1B-like
D8B = draft("8B", 8.0e9, 32, 8, 128)       # Llama-3-8B-like
EAGLE = draft("eagle-head", 0.9e9, 1, 8, 128)   # one decoder layer's worth, reuses target features (ASSUMED shape)

def step_time(m, batch, ctx, tokens_per_seq=1, mfu=0.5, overhead=1.0e-3):
    """One forward. Memory time reads the weights once plus every sequence's KV cache; compute time covers
    batch*tokens_per_seq tokens. Roofline: the larger of the two, plus fixed overhead and tensor-parallel comm."""
    mem = (m.weights_per_gpu() + batch * ctx * m.kv_per_gpu_token()) / HBM_BW
    comp = 2.0 * m.params * batch * tokens_per_seq / (m.gpus * m.peak * mfu)
    return max(mem, comp) + overhead + m.comm_s()

def crossover_tokens(m, batch, ctx, mfu=0.5):
    """Tokens in flight at which compute time equals memory time (the roofline ridge) for this batch and context."""
    mem = (m.weights_per_gpu() + batch * ctx * m.kv_per_gpu_token()) / HBM_BW
    return mem / (2.0 * m.params / (m.gpus * m.peak * mfu))

def kv_budget_tokens(target, draft_on_gpu0=None):
    """Equivalent target-KV token capacity on the worst-loaded rank.

    The draft's weights AND KV occupy GPU0. Express their combined KV cost
    in target-token equivalents so max_batch remains floor(capacity/context).
    Speculative temporary slots/metadata must fit the explicit RESERVE; this
    remains a modeled upper bound, not an engine-reported admission limit.
    """
    extra = draft_on_gpu0.weights_per_gpu() if draft_on_gpu0 else 0.0
    free = MEM_UTIL * GPU_MEM - target.weights_per_gpu() - extra - RESERVE
    combined_kv = target.kv_per_gpu_token() + (draft_on_gpu0.kv_per_gpu_token() if draft_on_gpu0 else 0.0)
    return max(0.0, free / combined_kv)


def max_batch(target, ctx, draft_m=None):
    if ctx <= 0:
        raise ValueError("context length must be positive")
    return int(kv_budget_tokens(target, draft_m) // ctx)


def expected_tokens(alpha, K):
    """E[tokens per verify pass], i.i.d. per-position acceptance alpha."""
    return K + 1.0 if alpha >= 1.0 else (1.0 - alpha ** (K + 1)) / (1.0 - alpha)

def expected_tokens_positional(alphas):
    """E = 1 + sum_i prod_{j<=i} alpha_j   (alphas[i] is the acceptance prob at position i+1 GIVEN earlier accepted)."""
    e, p = 1.0, 1.0
    for a in alphas:
        p *= a; e += p
    return e

def cycle_time(target, dm, K, batch, ctx, mfu=0.5):
    td = K * step_time(dm, batch, ctx, 1, mfu)                 # K sequential draft steps
    tv = step_time(target, batch, ctx, K + 1, mfu)             # one verify pass over K+1 tokens per sequence
    return td + tv

def speedup(target, dm, alpha, K, batch, ctx, mfu=0.5):
    base = step_time(target, batch, ctx, 1, mfu)
    return expected_tokens(alpha, K) * base / cycle_time(target, dm, K, batch, ctx, mfu)

def best_K(target, dm, alpha, batch, ctx, Kmax=12, mfu=0.5):
    best = max(range(1, Kmax + 1), key=lambda k: speedup(target, dm, alpha, k, batch, ctx, mfu))
    return best, speedup(target, dm, alpha, best, batch, ctx, mfu)

def simulate_tokens_per_pass(alpha, K, n=200000, seed=0):
    rng = random.Random(seed); tot = 0
    for _ in range(n):
        acc = 0
        while acc < K and rng.random() < alpha: acc += 1
        tot += acc + 1
    return tot / n

def simulate_positional(alphas, n=200000, seed=0):
    rng = random.Random(seed); tot = 0
    for _ in range(n):
        acc = 0
        for a in alphas:
            if rng.random() < a: acc += 1
            else: break
        tot += acc + 1
    return tot / n

def acceptance(p, q): return sum(min(a, b) for a, b in zip(p, q))
def softmax(z, T):
    m = max(z); e = [math.exp((x - m) / T) for x in z]; s = sum(e); return [x / s for x in e]

def alpha_vs_temperature(seed=1, V=50, noise=0.8, temps=(0.2, 0.5, 1.0, 1.5)):
    """Synthetic: draft logits = target logits + noise. The TREND is a property of this noise model, not of real models."""
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
    r = []
    def chk(name, ok, info=""):
        r.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")
    t = llama70(); t8 = llama70(1, PEAK_FP8)
    chk("70B BF16 TP2 weights-only decode floor = 20.9 ms", abs(t.weights_per_gpu() / HBM_BW * 1e3 - 20.9) < 0.05)
    chk("KV bytes/token for a Llama-3-70B shape = 327,680 total, 163,840 per GPU at TP2",
        t.kv_bytes_per_token == 327680 and t.kv_per_gpu_token() == 163840)
    chk("80 layers x 2 all-reduces x 12 us = 1.92 ms TP communication", abs(t.comm_s() - 1.92e-3) < 1e-9)
    x = crossover_tokens(t, 1, 0)
    chk("compute/memory crossover with no KV is 147.6 tokens (NOT 295: MFU 0.5 halves it)", abs(x - 147.6) < 0.5, f"({x:.1f})")
    chk("crossover grows with KV traffic: 24 seqs x 1024 ctx gives a higher ridge",
        crossover_tokens(t, 24, 1024) > crossover_tokens(t, 1, 1024))
    # memory feasibility
    mb2k = max_batch(t, 2048, D1B)
    chk("BF16 TP2 + 1B draft: max batch at 2k context is small (KV-limited)", 1 <= mb2k <= 20, f"(max batch {mb2k})")
    chk("BF16 TP2 + 8B draft colocated does not fit (70 GB + 16 GB > 77 GB budget)", kv_budget_tokens(t, D8B) == 0.0)
    free = MEM_UTIL * GPU_MEM - 35e9 - 16e9 - RESERVE                 # independent hand computation
    chk("FP8 TP2 + 8B draft colocated fits; max batch at 2k context equals the hand computation",
        max_batch(t8, 2048, D8B) == int(free / (163840 + D8B.kv_per_gpu_token()) // 2048) and max_batch(t8, 2048, D8B) > 0,
        f"(max batch {max_batch(t8, 2048, D8B)})")
    # math identities
    errs = [abs(simulate_tokens_per_pass(a, K) - expected_tokens(a, K)) for a in (0.3, .6, .8, .9) for K in (1, 3, 5, 8)]
    chk("E[tokens] closed form agrees with simulating the accept/reject chain (a sanity check of the formula)", max(errs) < 0.02, f"(max err {max(errs):.4f})")
    chk("limits: alpha=0 gives 1 token, alpha=1 gives K+1", expected_tokens(0, 5) == 1.0 and expected_tokens(1, 5) == 6.0)
    chk("positional formula equals the i.i.d. formula for constant alpha",
        abs(expected_tokens_positional([0.7] * 5) - expected_tokens(0.7, 5)) < 1e-12)
    dec = [0.9, 0.8, 0.7, 0.6, 0.5]
    chk("positional formula matches simulation for decaying alpha", abs(expected_tokens_positional(dec) - simulate_positional(dec)) < 0.02,
        f"({expected_tokens_positional(dec):.3f})")
    chk("acceptance(p,p)=1; disjoint support = 0", abs(acceptance([.5, .5, 0], [.5, .5, 0]) - 1) < 1e-12 and acceptance([.5, .5, 0], [0, 0, 1]) == 0)
    # speedup properties
    chk("K=0 gives speedup exactly 1.0", abs(speedup(t, D1B, 0.8, 0, 1, 1024) - 1.0) < 1e-12)
    s = [speedup(t, D1B, a, 4, 1, 1024) for a in (0.2, 0.4, 0.6, 0.8, 0.95)]
    chk("speedup strictly increases with alpha", all(a < b for a, b in zip(s, s[1:])))
    ks = [best_K(t, D1B, a, 1, 1024)[0] for a in (0.3, 0.5, 0.7, 0.9)]
    chk("best K nondecreasing in alpha", all(a <= b for a, b in zip(ks, ks[1:])), f"({ks})")
    sb = [speedup(t8, D1B, 0.8, 4, b, 256) for b in (16, 32, 64, 128, 256, 512)]
    chk("FP8 TP2, ctx 256, K=4: speedup nonincreasing in batch (past the crossover it must fall)",
        all(a >= b - 1e-9 for a, b in zip(sb, sb[1:])), f"({[round(v, 2) for v in sb]})")
    chk("a draft as expensive as the target loses to plain decoding", speedup(t, t, 0.7, 4, 1, 1024) < 1.0)
    chk("for batch 1 the 1B draft beats the 8B draft at equal alpha (cheaper cycle)",
        speedup(t8, D1B, 0.8, 4, 1, 1024) > speedup(t8, D8B, 0.8, 4, 1, 1024))
    return all(r)

def fmt(t, dm, alpha, B, ctx, mb):
    if B > mb: return "n/a (KV)  "
    k, s = best_K(t, dm, alpha, B, ctx)
    return f"{s:4.2f}x K={k:<2d} "

def tables():
    t, t8 = llama70(), llama70(1, PEAK_FP8)
    print("\n== A. Target 70B BF16 on TP2. Draft 1B colocated on GPU0. ctx = 1024 (ASSUMED alpha, mfu 0.5, overhead 1 ms) ==")
    mb = max_batch(t, 1024, D1B); print(f"   KV budget {kv_budget_tokens(t, D1B):,.0f} tokens -> max batch {mb} at ctx 1024")
    Bs = (1, 4, 8, 16, 24)
    print("   alpha   " + "".join(f"B={b:<10d}" for b in Bs))
    for a in (0.5, 0.7, 0.8, 0.9):
        print(f"   {a:.1f}     " + "".join(fmt(t, D1B, a, b, 1024, mb) for b in Bs))
    print("   base step (ms): " + ", ".join(f"B={b}: {step_time(t, b, 1024)*1e3:.1f}" for b in Bs if b <= mb))
    for dm, nm in ((D1B, "1B draft"), (D8B, "8B draft"), (EAGLE, "EAGLE-style head")):
        mb = max_batch(t8, 2048, dm)
        print(f"\n== B. Target 70B FP8 on TP2. {nm} colocated. ctx = 2048. KV budget {kv_budget_tokens(t8, dm):,.0f} tokens -> max batch {mb} ==")
        Bs = (1, 8, 32, 64, 96, 128)
        print("   alpha   " + "".join(f"B={b:<10d}" for b in Bs))
        for a in (0.5, 0.7, 0.8, 0.9):
            print(f"   {a:.1f}     " + "".join(fmt(t8, dm, a, b, 2048, mb) for b in Bs))
        print("   base step (ms): " + ", ".join(f"B={b}: {step_time(t8, b, 2048)*1e3:.1f}" for b in Bs if b <= mb))
        print("   ridge (tokens in flight) : " + ", ".join(f"B={b}: {crossover_tokens(t8, b, 2048):.0f}" for b in Bs if b <= mb))
    print("\n== C. alpha vs temperature (synthetic noise model; trend is NOT a property of real models) ==")
    for T, a in alpha_vs_temperature().items(): print(f"   T={T:<4}: alpha={a:.3f}")

if __name__ == "__main__":
    print("self-tests:"); ok = selftests(); tables()
    print("\nALL PASS" if ok else "\nFAILURES"); raise SystemExit(0 if ok else 1)
