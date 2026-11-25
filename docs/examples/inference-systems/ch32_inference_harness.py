"""Chapter 32: inference-engine anatomy and a measurement harness (CPU simulation).

Run:  python3 ch32_inference_harness.py          (the chapter's numbers)
      python3 ch32_inference_harness.py --test   (self-tests)
      python3 ch32_inference_harness.py --extras (batch-cap and prefill-interference experiments)
The engine below is a SIMULATION driven by an illustrative cost model. Its constants are parameters you
replace with measurements from a real server; none of its timings is an H100 measurement.
"""
import math
import random
from dataclasses import dataclass

# ------------------------------------------------------------- datasheet-level constants (check against your own)
H100_HBM_BW = 3.35e12            # bytes/s, H100 SXM HBM3 (NVIDIA datasheet value)
LLAMA70B = dict(layers=80, kv_heads=8, head_dim=128, params=70e9)


def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_elem=2):
    return 2 * layers * kv_heads * head_dim * bytes_per_elem


def decode_floor_ms(params, bytes_per_param, n_gpus, bw=H100_HBM_BW):
    """Lower bound on one decode step at batch 1: every weight byte is read once, spread over n GPUs."""
    return params * bytes_per_param / (n_gpus * bw) * 1e3


# ------------------------------------------------------------- workload
@dataclass
class Req:
    rid: int
    arrival: float
    prompt: int
    output: int
    first_token: float = math.nan
    finish: float = math.nan
    generated: int = 0


def make_workload(n, rate, seed, prompt_med=512, out_med=128, sigma=0.7):
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate)                                   # Poisson arrivals
        p = max(1, int(rng.lognormvariate(math.log(prompt_med), sigma)))
        o = max(1, int(rng.lognormvariate(math.log(out_med), sigma)))
        reqs.append(Req(i, t, p, o))
    return reqs


# ------------------------------------------------------------- simulated continuous-batching engine
@dataclass
class CostModel:
    base: float = 0.020              # s, fixed cost of a decode iteration (weight read)
    per_seq: float = 0.0002          # s per running sequence
    per_kv_token: float = 1.0e-8     # s per cached token read
    prefill_tps: float = 40000.0     # prompt tokens per second


def simulate(reqs, max_batch=64, cost=CostModel()):
    reqs = [Req(r.rid, r.arrival, r.prompt, r.output) for r in reqs]
    waiting, running, now, i = [], [], 0.0, 0
    area, last_t = 0.0, 0.0                                           # for Little's law: integral of in-system count
    done = 0
    while done < len(reqs):
        while i < len(reqs) and reqs[i].arrival <= now:
            waiting.append(reqs[i]); i += 1
        if not running and not waiting:
            now = reqs[i].arrival                                      # idle: jump to next arrival
            continue
        admit = []
        while waiting and len(running) + len(admit) < max_batch:
            admit.append(waiting.pop(0))
        prefill_t = sum(r.prompt for r in admit) / cost.prefill_tps
        running += admit
        kv = sum(r.prompt + r.generated for r in running)
        step_t = cost.base + cost.per_seq * len(running) + cost.per_kv_token * kv
        in_system = len(running) + len(waiting)
        dt = prefill_t + step_t
        area += in_system * dt
        now += dt
        for r in list(running):
            r.generated += 1
            if math.isnan(r.first_token):
                r.first_token = now
            if r.generated >= r.output:
                r.finish = now; running.remove(r); done += 1
    return reqs, now, area


# ------------------------------------------------------------- metrics
def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(reqs, wall, gpu_hour_usd=None, n_gpus=1, slo_ttft=2.0, slo_tpot=0.1):
    ttft = [r.first_token - r.arrival for r in reqs]
    tpot = [(r.finish - r.first_token) / max(1, r.output - 1) for r in reqs]
    toks = sum(r.output for r in reqs)
    good = [r for r, a, b in zip(reqs, ttft, tpot) if a <= slo_ttft and b <= slo_tpot]
    out = dict(ttft_p50=pct(ttft, .5), ttft_p99=pct(ttft, .99), tpot_p50=pct(tpot, .5), tpot_p99=pct(tpot, .99),
               throughput_tok_s=toks / wall, goodput_req_s=len(good) / wall, slo_attainment=len(good) / len(reqs))
    if gpu_hour_usd:
        out["usd_per_million_tokens"] = gpu_hour_usd * n_gpus * (wall / 3600) / (toks / 1e6)
    return out


# ------------------------------------------------------------- self-tests (properties, not just one hand case)
def run_tests():
    """Twenty-one checks. Each is an invariant the harness must keep; none depends on the cost constants being realistic."""
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    # arithmetic
    check("kv bytes/token for the 70B shape is 327680", kv_bytes_per_token(80, 8, 128) == 327680)
    check("kv bytes scale linearly with layers", kv_bytes_per_token(160, 8, 128) == 2 * kv_bytes_per_token(80, 8, 128))
    check("decode floor halves when GPUs double", abs(decode_floor_ms(70e9, 2, 2) * 2 - decode_floor_ms(70e9, 2, 1)) < 1e-9)
    check("decode floor is linear in bytes/param", abs(decode_floor_ms(70e9, 1, 2) * 2 - decode_floor_ms(70e9, 2, 2)) < 1e-9)
    # percentile
    check("pct: q=0 is min, q=1 is max", pct([3, 1, 2], 0) == 1 and pct([3, 1, 2], 1) == 3)
    check("pct: median of [1,2,3,4] is 2.5", pct([1, 2, 3, 4], .5) == 2.5)
    check("pct: one element", pct([7.0], .99) == 7.0)
    check("pct: monotone in q", all(pct(list(range(20)), a / 10) <= pct(list(range(20)), (a + 1) / 10) for a in range(10)))
    # workload
    w1, w2, w3 = make_workload(50, 4.0, 7), make_workload(50, 4.0, 7), make_workload(50, 4.0, 8)
    check("workload: same seed gives identical requests", [(r.arrival, r.prompt, r.output) for r in w1] == [(r.arrival, r.prompt, r.output) for r in w2])
    check("workload: different seed differs", [r.arrival for r in w1] != [r.arrival for r in w3])
    check("workload: arrivals non-decreasing, lengths >= 1",
          all(a.arrival <= b.arrival for a, b in zip(w1, w1[1:])) and all(r.prompt >= 1 and r.output >= 1 for r in w1))
    # engine
    src = make_workload(120, 6.0, 3)
    before = [(r.first_token, r.finish, r.generated) for r in src]
    reqs, wall, _ = simulate(src)
    check("simulate: does not mutate its input", before == [(r.first_token, r.finish, r.generated) for r in src])
    check("simulate: every request finishes with exactly `output` tokens, ordered times",
          all(r.generated == r.output and r.arrival <= r.first_token <= r.finish <= wall + 1e-9 for r in reqs))
    one, _, _ = simulate(make_workload(30, 50.0, 4), max_batch=1)
    check("max_batch=1 serves strictly one at a time (no overlap)", all(b.first_token > a.finish for a, b in zip(one, one[1:])))
    lo, wlo, _ = simulate(make_workload(200, 2.0, 5))
    hi, whi, _ = simulate(make_workload(200, 20.0, 5))
    check("same lengths, 10x the arrival rate: p99 TTFT does not fall",
          summarize(hi, whi)["ttft_p99"] >= summarize(lo, wlo)["ttft_p99"])
    # metrics
    loose, tight = summarize(reqs, wall, slo_ttft=5.0, slo_tpot=1.0), summarize(reqs, wall, slo_ttft=0.05, slo_tpot=0.02)
    toy = [Req(0, 0.0, 10, 5, first_token=0.5, finish=0.9), Req(1, 1.0, 10, 3, first_token=1.2, finish=1.4)]
    st = summarize(toy, 1.4, slo_ttft=0.4, slo_tpot=0.15)
    check("hand case: TPOT = (finish-first)/(output-1) = 0.10 for both, TTFT p50 = 0.35, goodput = 1/1.4",
          abs(st["tpot_p50"] - 0.10) < 1e-12 and abs(st["ttft_p50"] - 0.35) < 1e-12 and abs(st["goodput_req_s"] - 1 / 1.4) < 1e-12)
    check("tighter SLO never raises attainment", tight["slo_attainment"] <= loose["slo_attainment"])
    one_tok = [Req(0, 0.0, 5, 1, first_token=0.3, finish=0.3)]
    check("a 1-token reply does not divide by zero", summarize(one_tok, 0.3)["tpot_p50"] == 0.0)
    cost = summarize([Req(0, 0, 1, 1000, first_token=1.0, finish=2.0)], 3600.0, gpu_hour_usd=4.0, n_gpus=2)
    check("cost: $4 x 2 GPUs x 1 h / 1000 tokens = $8000 per million", abs(cost["usd_per_million_tokens"] - 8000.0) < 1e-6)
    ok = True
    for seed in range(5):
        rr, ww, aa = simulate(make_workload(300, 4.0, seed))
        L, lamW = aa / ww, len(rr) / ww * sum(r.finish - r.arrival for r in rr) / len(rr)
        ok &= abs(L - lamW) / L < 0.02
    check("Little's law within 2% on five seeds", ok)
    try:
        pct([], .5); empty_raises = False
    except IndexError:
        empty_raises = True
    check("pct([]) raises IndexError (documented failure mode)", empty_raises)
    for name, good in results:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
    print(f"{sum(g for _, g in results)}/{len(results)} self-tests pass")
    return all(g for _, g in results)


def extras():
    """Two experiments the chapter discusses: the batch cap, and prefill interference with running decodes."""
    print("Experiment A: batch cap at 12 req/s (300 requests, seed 2)")
    print("max_batch  ttft_p50  ttft_p99  tpot_p50  tput(tok/s)  slo_ok")
    for mb in (4, 8, 16, 32, 64, 128):
        r, w, _ = simulate(make_workload(300, rate=12, seed=2), max_batch=mb)
        m = summarize(r, w)
        print(f"{mb:>9}  {m['ttft_p50']:8.3f}  {m['ttft_p99']:8.3f}  {m['tpot_p50']:8.4f}  {m['throughput_tok_s']:11.0f}  {m['slo_attainment']:6.2f}")
    print("\nExperiment B: prefill interference at 4 req/s (300 requests, seed 2, output median 128)")
    print("prompt_median  ttft_p50  tpot_p50  tpot_p99  tput(tok/s)")
    for pm in (128, 512, 2048, 8192):
        r, w, _ = simulate(make_workload(300, rate=4, seed=2, prompt_med=pm))
        m = summarize(r, w)
        print(f"{pm:>13}  {m['ttft_p50']:8.3f}  {m['tpot_p50']:8.4f}  {m['tpot_p99']:8.4f}  {m['throughput_tok_s']:11.0f}")


def main():
    kv = kv_bytes_per_token(**{k: LLAMA70B[k] for k in ("layers", "kv_heads", "head_dim")})
    print(f"KV bytes/token (Llama-3-70B-shaped, BF16) = {kv} ({kv/1024:.0f} KiB); 4096-token prompt = {kv*4096/1e9:.2f} GB")
    print(f"decode floor, 70B BF16 on 2 H100: {decode_floor_ms(70e9, 2, 2):.1f} ms/token at batch 1 "
          f"(weights only; ignores KV reads, all-reduce and launch cost)")
    print(f"decode floor, 70B FP8  on 1 H100: {decode_floor_ms(70e9, 1, 1):.1f} ms/token at batch 1")

    # metric function against a hand-computed case
    toy = [Req(0, 0.0, 10, 5, first_token=0.5, finish=0.9), Req(1, 1.0, 10, 3, first_token=1.2, finish=1.4)]
    s = summarize(toy, wall=1.4, slo_ttft=0.4, slo_tpot=0.15)
    print(f"hand-check: ttft_p50={s['ttft_p50']:.3f} (expect 0.350)  tpot_p50={s['tpot_p50']:.3f} (expect 0.100)  "
          f"goodput={s['goodput_req_s']:.3f} req/s (expect 0.714: only req 1 meets TTFT<=0.4)")

    # Little's law check: L = lambda * W
    reqs, wall, area = simulate(make_workload(400, rate=4.0, seed=1))
    lam = len(reqs) / wall
    W = sum(r.finish - r.arrival for r in reqs) / len(reqs)
    L = area / wall
    print(f"Little's law: L={L:.3f}  lambda*W={lam*W:.3f}  rel.diff={abs(L-lam*W)/L:.3%}")

    # load sweep (simulated, illustrative cost model)
    print("\nrate(req/s)  ttft_p50  ttft_p99  tpot_p50  tput(tok/s)  slo_ok  $/Mtok(@$4/GPU-h, 2 GPUs)")
    for rate in (1, 4, 8, 12, 16):
        r, w, _ = simulate(make_workload(300, rate=rate, seed=2))
        m = summarize(r, w, gpu_hour_usd=4.0, n_gpus=2)
        print(f"{rate:>10}  {m['ttft_p50']:8.3f}  {m['ttft_p99']:8.3f}  {m['tpot_p50']:8.4f}  "
              f"{m['throughput_tok_s']:11.0f}  {m['slo_attainment']:6.2f}  {m['usd_per_million_tokens']:8.2f}")


if __name__ == "__main__":
    import sys
    if "--extras" in sys.argv:
        extras(); sys.exit(0)
    if "--test" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    main()
