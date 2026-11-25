#!/usr/bin/env python3
"""Chapter 41 -- Resilience and edge fallback (CPU-only, stdlib only).

Parts
  1. CircuitBreaker, RetryBudget, full-jitter backoff   (the building blocks)
  2. Per-client simulation: five outage shapes x six policies P0..P5
  3. Fleet simulation: a capacity dip, retries, and metastable collapse
  4. Edge feasibility arithmetic (bandwidth bound, WebGPU buffer limits)

Every number printed by `experiments()` comes from the ASSUMED models in the
constants below. Nothing here touches WebGPU, WebLLM or MLX.

    python3 ch41_edge_fallback.py            # tests + experiments
    python3 ch41_edge_fallback.py tests      # tests only
"""
import heapq
import math
import sys
from collections import deque
from dataclasses import dataclass
from random import Random

# ----------------------------------------------------------------------------
# Assumed model constants (not measurements)
# ----------------------------------------------------------------------------
TTFT = 0.5           # remote time to first token, s
REMOTE_RATE = 60.0   # remote decode, tokens/s
TTFT_TIMEOUT = 3.0   # client gives up waiting for a first token after this
DEADLINE = 30.0      # total budget per logical request, s
OUTAGE = (60.0, 180.0)
HORIZON = 300.0

EDGE_PREFILL = 300.0  # tokens/s
EDGE_DECODE = 25.0    # tokens/s
EDGE_CTX = 2048       # context window, tokens
EDGE_QUEUE = 2        # waiting slots behind the one running request


# ----------------------------------------------------------------------------
# 1. Building blocks
# ----------------------------------------------------------------------------
class CircuitBreaker:
    """closed -> open -> half_open -> closed.

    allow(now) returns (mode, gen) with mode in {"closed", "probe", None}.
    record(gen, ok, now) ignores results whose gen is stale, so a slow call
    that started before the breaker opened cannot close it again.
    """

    def __init__(self, threshold=3, open_for=10.0):
        self.threshold = threshold
        self.open_for = open_for
        self.state = "closed"
        self.fails = 0
        self.opened_at = 0.0
        self.probe_at = 0.0
        self.gen = 0

    def allow(self, now):
        if self.state == "closed":
            return "closed", self.gen
        if self.state == "open":
            if now - self.opened_at >= self.open_for:
                self.state = "half_open"
                self.gen += 1
                self.probe_at = now
                return "probe", self.gen
            return None, self.gen
        # half_open: exactly one probe in flight; a lost probe is replaced
        if now - self.probe_at >= self.open_for:
            self.gen += 1
            self.probe_at = now
            return "probe", self.gen
        return None, self.gen

    def record(self, gen, ok, now):
        if gen != self.gen:
            return False
        if self.state == "closed":
            if ok:
                self.fails = 0
            else:
                self.fails += 1
                if self.fails >= self.threshold:
                    self._open(now)
        elif self.state == "half_open":
            if ok:
                self.state = "closed"
                self.fails = 0
                self.gen += 1
            else:
                self._open(now)
        return True

    def _open(self, now):
        self.state = "open"
        self.opened_at = now
        self.gen += 1


class RetryBudget:
    """Token bucket: every request earns `ratio` tokens, a retry costs one."""

    def __init__(self, ratio=0.1, cap=10.0, initial=None):
        self.ratio = ratio
        self.cap = cap
        self.initial = cap if initial is None else initial
        self.tokens = self.initial
        self.requests = 0
        self.granted = 0

    def on_request(self):
        self.requests += 1
        self.tokens = min(self.cap, self.tokens + self.ratio)

    def try_retry(self):
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            self.granted += 1
            return True
        return False


def backoff(rng, attempt, base=0.2, cap=2.0):
    """Full jitter: uniform(0, min(cap, base * 2**attempt))."""
    return rng.uniform(0.0, min(cap, base * 2 ** attempt))


# ----------------------------------------------------------------------------
# 2. Per-client simulation
# ----------------------------------------------------------------------------
@dataclass
class Req:
    t: float
    prompt: int
    out: int


def make_requests(seed, horizon=HORIZON, mean_gap=12.0):
    rng = Random(seed)
    t, reqs = 0.0, []
    while True:
        t += rng.expovariate(1.0 / mean_gap)
        if t >= horizon:
            return reqs
        prompt = 3000 if rng.random() < 0.10 else rng.randint(100, 800)
        reqs.append(Req(t, prompt, rng.randint(80, 300)))


def _u(seed, idx, attempt):
    return Random(f"{seed}-{idx}-{attempt}").random()


def in_outage(scn, t):
    if scn == "none" or not (OUTAGE[0] <= t < OUTAGE[1]):
        return False
    if scn == "flapping":
        return int((t - OUTAGE[0]) // 10) % 2 == 0
    return True


def remote_attempt(scn, t, out, have, u):
    """-> (duration, status, tokens_held_after). status: ok|refused|timeout|dropped."""
    remaining = out - have
    full = TTFT + remaining / REMOTE_RATE
    if not in_outage(scn, t):
        return full, "ok", out
    if scn == "refused":
        return 0.05, "refused", have
    if scn in ("hang", "flapping"):
        return TTFT_TIMEOUT, "timeout", have
    if scn == "brownout":
        return (TTFT_TIMEOUT, "timeout", have) if u < 0.5 else (full, "ok", out)
    if scn == "middrop":
        if remaining <= 1:
            return full, "ok", out
        got = remaining // 2
        return TTFT + got / REMOTE_RATE, "dropped", have + got
    raise ValueError(scn)


@dataclass(frozen=True)
class Policy:
    name: str
    attempts: int = 1
    backoff: bool = False
    breaker: bool = False
    edge: bool = False
    resume: bool = False


POLICIES = [
    Policy("P0 single try"),
    Policy("P1 retry x3, no backoff", attempts=4),
    Policy("P2 +jittered backoff", attempts=4, backoff=True),
    Policy("P3 +circuit breaker", attempts=4, backoff=True, breaker=True),
    Policy("P4 +edge fallback", attempts=4, backoff=True, breaker=True, edge=True),
    Policy("P5 +mid-stream resume", attempts=4, backoff=True, breaker=True,
           edge=True, resume=True),
]


class ClientState:
    def __init__(self):
        self.br = CircuitBreaker()
        self.edge_jobs = []
        self.edge_max_active = 0
        self.edge_ctx_violations = 0
        self.remote_attempts = 0


def edge_serve(state, t, req, have):
    """Returns finish time, or None if the edge cannot take the request."""
    eff_prompt = req.prompt + have
    remaining = req.out - have
    if eff_prompt + remaining > EDGE_CTX:
        return None
    active = [f for f in state.edge_jobs if f > t]
    if len(active) >= 1 + EDGE_QUEUE:
        return None
    start = max([t] + active)
    finish = start + eff_prompt / EDGE_PREFILL + remaining / EDGE_DECODE
    state.edge_jobs.append(finish)
    state.edge_max_active = max(state.edge_max_active, len(active) + 1)
    if eff_prompt + remaining > EDGE_CTX:
        state.edge_ctx_violations += 1
    return finish


def serve(req, idx, pol, scn, state, seed, rng):
    """-> (ok, latency, source) with source in remote|edge|None."""
    t, deadline, have = req.t, req.t + DEADLINE, 0
    for k in range(pol.attempts):
        if t >= deadline:
            break
        gen = None
        if pol.breaker:
            mode, gen = state.br.allow(t)
            if mode is None:
                break
        dur, status, held = remote_attempt(scn, t, req.out, have,
                                           _u(seed, idx, k))
        state.remote_attempts += 1
        t_end = t + dur
        if pol.breaker:
            state.br.record(gen, status == "ok", t_end)
        if status == "ok":
            return (t_end <= deadline), t_end - req.t, "remote"
        if pol.resume:
            have = held
        t = t_end + (backoff(rng, k) if pol.backoff else 0.0)
    if pol.edge and t < deadline:
        fin = edge_serve(state, t, req, have if pol.resume else 0)
        if fin is not None and fin <= deadline:
            return True, fin - req.t, "edge"
    return False, None, None


def run_client(pol, scn, seed):
    reqs = make_requests(seed)
    state, rng = ClientState(), Random(seed + 7)
    res = [serve(r, i, pol, scn, state, seed, rng) for i, r in enumerate(reqs)]
    n = len(reqs)
    lats = sorted(l for ok, l, s in res if ok)
    p95 = lats[min(len(lats) - 1, int(0.95 * len(lats)))] if lats else float("nan")
    return {
        "n": n,
        "success": sum(1 for ok, _, _ in res if ok) / n,
        "edge_share": sum(1 for _, _, s in res if s == "edge") / n,
        "p95": p95,
        "remote_per_req": state.remote_attempts / n,
        "state": state,
    }


# ----------------------------------------------------------------------------
# 3. Fleet simulation: capacity dip and retry storm
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class FleetPolicy:
    name: str
    retries: bool = False
    jitter: bool = False
    budget: bool = False
    shed: bool = False


FLEET_POLICIES = [
    FleetPolicy("F0 no retries"),
    FleetPolicy("F1 immediate retries x3", retries=True),
    FleetPolicy("F2 +jittered backoff", retries=True, jitter=True),
    FleetPolicy("F3 +retry budget 10%", retries=True, jitter=True, budget=True),
    FleetPolicy("F4 +expired-request shedding", retries=True, jitter=True,
                budget=True, shed=True),
    FleetPolicy("F5 shedding only", shed=True),
]


def simulate_fleet(pol, seed=0, horizon=150.0, lam=70.0, mu=100.0,
                   dip=(20.0, 50.0), dip_frac=0.3, timeout=5.0,
                   max_retries=3, dt=0.05, window=5.0):
    rng = Random(seed)
    arrivals, t = [], 0.0
    while True:
        t += rng.expovariate(lam)
        if t >= horizon:
            break
        arrivals.append(t)
    ai = 0
    attempts = {}        # id -> [logical, arrive, state, attempt_no]
    queue = deque()
    timeouts, retries = [], []   # heaps
    logical_ok, logical_fail = set(), set()
    budget = RetryBudget(0.1, cap=1e9, initial=50.0) if pol.budget else None
    nid, credit = 0, 0.0
    nwin = int(horizon / window)
    good = [0] * nwin
    wasted = shed_n = served = 0
    peak_q = 0
    steps = int(horizon / dt)
    for s in range(steps):
        now = s * dt
        end = now + dt

        def enqueue(logical, attempt_no, at):
            nonlocal nid
            attempts[nid] = [logical, at, "queued", attempt_no]
            queue.append(nid)
            heapq.heappush(timeouts, (at + timeout, nid))
            nid += 1

        while ai < len(arrivals) and arrivals[ai] < end:
            if budget:
                budget.on_request()
            enqueue(ai, 0, arrivals[ai])
            ai += 1
        while retries and retries[0][0] < end:
            due, logical, attempt_no = heapq.heappop(retries)
            enqueue(logical, attempt_no, due)
        while timeouts and timeouts[0][0] < end:
            dl, a = heapq.heappop(timeouts)
            logical, arrive, st, k = attempts[a]
            if st != "queued":
                continue
            attempts[a][2] = "abandoned"
            if (pol.retries and k < max_retries and logical not in logical_ok
                    and (budget is None or budget.try_retry())):
                delay = 0.0
                if pol.jitter:
                    delay = rng.uniform(0.0, min(8.0, 1.0 * 2 ** k))
                heapq.heappush(retries, (dl + delay, logical, k + 1))
            else:
                if logical not in logical_ok:
                    logical_fail.add(logical)
        in_dip = dip[0] <= now < dip[1]
        credit += (mu * dip_frac if in_dip else mu) * dt
        while credit >= 1.0 and queue:
            a = queue.popleft()
            logical, arrive, st, k = attempts[a]
            if pol.shed and (now - arrive) >= timeout:
                shed_n += 1
                continue
            credit -= 1.0
            served += 1
            if st == "queued" and (now - arrive) < timeout:
                attempts[a][2] = "done"
                logical_ok.add(logical)
                logical_fail.discard(logical)
                good[min(nwin - 1, int(now / window))] += 1
            else:
                wasted += 1
        if not queue:
            credit = min(credit, 1.0)
        peak_q = max(peak_q, len(queue))
    goodput = [g / window for g in good]
    rec = None
    for w in range(int(dip[1] / window), nwin):
        if goodput[w] >= 0.9 * lam:
            rec = w * window - dip[1]
            break
    return {
        "goodput": goodput, "recovery_s": rec, "peak_queue": peak_q,
        "wasted": wasted, "shed": shed_n, "served": served,
        "arrivals": len(arrivals), "ok": len(logical_ok),
        "capacity_units": (mu * (horizon - (dip[1] - dip[0]))
                           + mu * dip_frac * (dip[1] - dip[0])),
        "budget": budget,
    }


# ----------------------------------------------------------------------------
# 4. Edge feasibility arithmetic
# ----------------------------------------------------------------------------
def edge_decode_bound(bandwidth_gb_s, params_b, bits):
    """Upper bound on decode tokens/s: bandwidth / bytes of weights per token."""
    return bandwidth_gb_s / (params_b * bits / 8.0)


def weight_gb(params_b, bits):
    return params_b * bits / 8.0


def min_bindings(model_bytes, limit_bytes):
    return math.ceil(model_bytes / limit_bytes)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def selftests():
    out = []

    def t(name):
        def deco(f):
            f()
            out.append(name)
        return deco

    @t("1 breaker opens after N consecutive failures, a success resets the count")
    def _():
        b = CircuitBreaker(3, 10)
        for ok in (False, False, True, False, False):
            m, g = b.allow(0)
            b.record(g, ok, 0)
        assert b.state == "closed" and b.fails == 2
        m, g = b.allow(0)
        b.record(g, False, 0)
        assert b.state == "open"

    @t("2 open rejects until open_for, then exactly one probe")
    def _():
        b = CircuitBreaker(1, 10)
        m, g = b.allow(0)
        b.record(g, False, 0)
        assert b.allow(5)[0] is None
        m, g = b.allow(10)
        assert m == "probe" and b.state == "half_open"
        assert b.allow(11)[0] is None          # second caller refused
        b.record(g, True, 12)
        assert b.state == "closed"

    @t("3 a failed probe reopens; a lost probe is replaced after open_for")
    def _():
        b = CircuitBreaker(1, 10)
        m, g = b.allow(0)
        b.record(g, False, 0)
        m, g = b.allow(10)
        b.record(g, False, 11)
        assert b.state == "open" and b.opened_at == 11
        m, g = b.allow(21)
        assert m == "probe"
        assert b.allow(25)[0] is None
        m2, g2 = b.allow(31)                   # probe never reported back
        assert m2 == "probe" and g2 != g

    @t("4 stale results are ignored (slow call cannot close an open breaker)")
    def _():
        b = CircuitBreaker(1, 10)
        m, slow_gen = b.allow(0)
        m, g = b.allow(0)
        b.record(g, False, 1)                  # opens
        assert b.state == "open"
        assert b.record(slow_gen, True, 2) is False
        assert b.state == "open"

    @t("5 retry budget: granted retries never exceed initial + ratio*requests")
    def _():
        rng = Random(1)
        for ratio in (0.0, 0.1, 0.5):
            rb = RetryBudget(ratio, cap=20, initial=5)
            for _ in range(2000):
                rb.on_request()
                for _ in range(rng.randint(0, 3)):
                    rb.try_retry()
            assert rb.granted <= rb.initial + ratio * rb.requests + 1e-9
            assert 0 <= rb.tokens <= rb.cap

    @t("6 full-jitter backoff stays in [0, min(cap, base*2^k)], mean about half")
    def _():
        rng = Random(2)
        for k in range(8):
            hi = min(2.0, 0.2 * 2 ** k)
            xs = [backoff(rng, k) for _ in range(4000)]
            assert min(xs) >= 0 and max(xs) <= hi
            assert abs(sum(xs) / len(xs) - hi / 2) < 0.05 * hi + 0.01

    @t("7 no outage: every policy succeeds on all requests, one remote try each")
    def _():
        for p in POLICIES:
            r = run_client(p, "none", 3)
            assert r["success"] == 1.0 and abs(r["remote_per_req"] - 1.0) < 1e-12

    @t("8 simulation is deterministic for a fixed seed")
    def _():
        a = run_client(POLICIES[5], "middrop", 4)
        b = run_client(POLICIES[5], "middrop", 4)
        assert all(a[k] == b[k] for k in ("success", "p95", "edge_share"))

    @t("9 refused: edge policy beats single try; breaker cuts remote load vs P1")
    def _():
        for seed in range(3):
            r0 = run_client(POLICIES[0], "refused", seed)
            r1 = run_client(POLICIES[1], "refused", seed)
            r3 = run_client(POLICIES[3], "refused", seed)
            r4 = run_client(POLICIES[4], "refused", seed)
            assert r4["success"] > r0["success"]
            assert r3["remote_per_req"] <= r1["remote_per_req"]

    @t("10 edge never serves beyond its context window, never exceeds its queue")
    def _():
        for scn in ("hang", "refused", "middrop"):
            for seed in range(3):
                r = run_client(POLICIES[5], scn, seed)
                assert r["state"].edge_ctx_violations == 0
                assert r["state"].edge_max_active <= 1 + EDGE_QUEUE

    @t("11 edge cannot rescue prompts longer than its context")
    def _():
        r = run_client(POLICIES[4], "hang", 5)
        reqs = make_requests(5)
        long_in_outage = sum(1 for q in reqs if q.prompt > EDGE_CTX
                             and in_outage("hang", q.t))
        assert long_in_outage > 0
        assert r["success"] < 1.0

    @t("12 resume lowers mean latency vs restarting after a mid-stream drop")
    def _():
        def mean_lat(p, seed):
            reqs = make_requests(seed)
            st, rng = ClientState(), Random(seed + 7)
            ls = [serve(q, i, p, "middrop", st, seed, rng)
                  for i, q in enumerate(reqs)]
            v = [l for ok, l, s in ls if ok and s == "edge"]
            return sum(v) / len(v)
        wins = sum(mean_lat(POLICIES[5], s) < mean_lat(POLICIES[4], s)
                   for s in range(5))
        assert wins == 5

    @t("13 fleet, no dip: goodput about equals arrival rate for every policy")
    def _():
        for p in FLEET_POLICIES:
            r = simulate_fleet(p, seed=1, dip=(1e9, 1e9 + 1))
            assert r["ok"] >= 0.99 * r["arrivals"]

    @t("14 fleet conservation: served <= capacity, ok <= arrivals")
    def _():
        for p in FLEET_POLICIES:
            r = simulate_fleet(p, seed=2)
            assert r["served"] <= r["capacity_units"] + 2
            assert r["ok"] <= r["arrivals"]
            assert sum(r["goodput"]) * 5 <= r["served"] + 1e-9

    @t("15 fleet: shedding recovers sooner than the naive-server retry storm")
    def _():
        naive = simulate_fleet(FLEET_POLICIES[1], seed=3)
        shed = simulate_fleet(FLEET_POLICIES[4], seed=3)
        assert shed["recovery_s"] is not None
        assert naive["recovery_s"] is None or shed["recovery_s"] < naive["recovery_s"]
        assert shed["wasted"] < naive["wasted"]

    @t("16 retry budget bounds fleet retries to initial + 10% of arrivals")
    def _():
        r = simulate_fleet(FLEET_POLICIES[3], seed=4)
        b = r["budget"]
        assert b.granted <= b.initial + 0.1 * b.requests + 1e-9

    @t("17 edge arithmetic hand checks")
    def _():
        assert abs(edge_decode_bound(200, 8, 4) - 50.0) < 1e-9
        assert abs(weight_gb(3, 4) - 1.5) < 1e-12
        assert min_bindings(1.5e9, 128 * 2 ** 20) == 12
        assert min_bindings(128 * 2 ** 20, 128 * 2 ** 20) == 1

    return out


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def fmt(v, spec=".1f", scale=1.0):
    v = [x * scale for x in v]
    return f"{sum(v) / len(v):{spec}} [{min(v):{spec}}-{max(v):{spec}}]"


def experiments():
    seeds = (0, 1, 2)
    print("\n=== Experiment 1: per-client outage shapes (mean [min-max], 3 seeds) ===")
    for scn in ("refused", "hang", "brownout", "flapping", "middrop"):
        print(f"\nscenario: {scn}")
        print(f"{'policy':26s} {'success %':>16s} {'p95 s':>16s} "
              f"{'edge %':>14s} {'remote/req':>14s}")
        for p in POLICIES:
            rs = [run_client(p, scn, s) for s in seeds]
            print(f"{p.name:26s} {fmt([r['success'] for r in rs], '.1f', 100):>16s} "
                  f"{fmt([r['p95'] for r in rs]):>16s} "
                  f"{fmt([r['edge_share'] for r in rs], '.1f', 100):>14s} "
                  f"{fmt([r['remote_per_req'] for r in rs], '.2f'):>14s}")

    print("\n=== Experiment 2: fleet retry storm "
          "(lam=70/s, mu=100/s, dip to 30% for 30 s, timeout 5 s) ===")
    print(f"{'policy':30s} {'recovery s':>14s} {'peak queue':>16s} "
          f"{'wasted':>16s} {'ok %':>14s}")
    for p in FLEET_POLICIES:
        rs = [simulate_fleet(p, seed=s) for s in seeds]
        rec = [r["recovery_s"] for r in rs]
        rec_s = ("never" if any(x is None for x in rec)
                 else fmt([float(x) for x in rec], ".0f"))
        print(f"{p.name:30s} {rec_s:>14s} "
              f"{fmt([r['peak_queue'] for r in rs], '.0f'):>16s} "
              f"{fmt([r['wasted'] for r in rs], '.0f'):>16s} "
              f"{fmt([r['ok'] / r['arrivals'] for r in rs], '.1f', 100):>14s}")
    print("\ngoodput per 5 s window, seed 0 (req/s):")
    for p in (FLEET_POLICIES[1], FLEET_POLICIES[4]):
        r = simulate_fleet(p, seed=0)
        print(f"{p.name:30s}", " ".join(f"{g:3.0f}" for g in r["goodput"]))

    print("\n=== Experiment 3: edge feasibility arithmetic (upper bounds) ===")
    print(f"{'device (assumed GB/s)':26s} {'model':12s} {'weights GB':>11s} "
          f"{'tok/s bound':>12s}")
    for dev, bw in (("phone 60", 60), ("laptop 100", 100), ("M-Pro 200", 200)):
        for params, bits in ((3, 4), (8, 4), (8, 8)):
            print(f"{dev:26s} {params}B @ {bits}-bit   "
                  f"{weight_gb(params, bits):11.1f} "
                  f"{edge_decode_bound(bw, params, bits):12.1f}")
    for gb in (1.5, 4.0):
        print(f"{gb} GB of weights needs >= "
              f"{min_bindings(gb * 1e9, 128 * 2 ** 20)} bindings at the 128 MiB default")


if __name__ == "__main__":
    names = selftests()
    for n in names:
        print("PASS", n)
    print(f"{len(names)} tests passed")
    if "tests" not in sys.argv[1:]:
        experiments()
