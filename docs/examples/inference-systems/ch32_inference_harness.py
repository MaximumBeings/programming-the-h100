"""Chapter 32: inference-engine anatomy and a measurement harness (CPU simulation).

Run:  python3 ch32_inference_harness.py
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
    main()
