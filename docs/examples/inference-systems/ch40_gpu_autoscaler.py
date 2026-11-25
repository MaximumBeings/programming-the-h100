#!/usr/bin/env python3
"""Chapter 40: queue-based GPU autoscaling (KEDA + HPA semantics) in a discrete-time simulator.

Run:  python3 ch40_gpu_autoscaler.py            # self-tests + experiments (CPU, numpy, ~1-2 min)
      python3 ch40_gpu_autoscaler.py --selftest

WHAT IS REAL: the scaling rules (modelled on the documented KEDA and Kubernetes HPA behaviour, see the chapter), the
queueing/billing accounting, the test and measurement method. WHAT IS INVENTED: the workload (arrival rates, service
times, cold-start times, GPU price). Nothing here was run on a cluster; every number is a property of the simulator.
Simplifications are listed in the chapter (no node provisioning delay, no image pull variance, no readiness gate beyond
a fixed cold start, no pod termination grace period beyond draining, FIFO queue, one GPU per replica).
"""
import math
import sys
import time
from collections import deque

import numpy as np


# ------------------------------------------------------------------ 1. the scaler (pure logic, no simulator inside)
class Scaler:
    """KEDA ScaledObject + HPA semantics.
    KEDA checks the trigger every `poll` seconds (default 30). KEDA alone handles 0 <-> 1: activate when the metric > 0;
    deactivate to 0 after the metric has been 0 for `cooldown` seconds (default 300) when min_replicas == 0.
    The HPA handles 1 <-> N every `hpa_period` seconds (default 15). With useCachedMetrics=false (the documented default) it
    reads a FRESH metric at each sync, so pollingInterval does not govern 1 <-> N; with cached_metrics=True it reads the
    value KEDA last cached at its poll. Either way:
        desired = ceil(metric_total / target_per_replica)            (external metric, AverageValue)
    no change if |ratio-1| <= tolerance (default 0.1) where ratio = (metric/current)/target;
    scale-up per period limited to max(100% of current, +4 pods) (selectPolicy Max, default);
    scale-down uses the maximum desired value over the last `down_window` seconds (default 300) and at most 100% per period.
    Result is clamped to [max(min_replicas,1) .. max_replicas] while the HPA is active."""
    def __init__(self, target, min_replicas=0, max_replicas=100, poll=30.0, cooldown=300.0, hpa_period=15.0,
                 tolerance=0.1, down_window=300.0, up_pods=4, up_percent=100.0, down_percent=100.0, cached_metrics=False):
        self.target, self.min_r, self.max_r = target, min_replicas, max_replicas
        self.poll, self.cooldown, self.hpa_period = poll, cooldown, hpa_period
        self.tol, self.down_window = tolerance, down_window
        self.up_pods, self.up_pct, self.down_pct = up_pods, up_percent, down_percent
        self.cached_metrics = cached_metrics       # KEDA useCachedMetrics (documented default: false -> the HPA gets a fresh value every sync)
        self.sample = 0.0
        self.idle_since = None
        self.history = deque()             # (time, desired) for the scale-down stabilization window

    def keda_poll(self, now, metric, current):
        """Called every `poll` seconds. Returns a new replica count if KEDA decides 0<->1, else None."""
        self.sample = metric
        if current == 0:
            self.idle_since = None
            return 1 if metric > 0 else None
        if metric > 0:
            self.idle_since = None
            return None
        if self.idle_since is None:
            self.idle_since = now
        if self.min_r == 0 and now - self.idle_since >= self.cooldown:
            return 0
        return None

    def hpa(self, now, current):
        """Called every `hpa_period` seconds while current >= 1. Returns the new replica count."""
        lo = max(self.min_r, 1)
        raw = math.ceil(self.sample / self.target) if self.sample > 0 else 0
        ratio = (self.sample / current) / self.target if current > 0 else math.inf
        desired = current if abs(ratio - 1.0) <= self.tol else raw
        desired = min(max(desired, lo), self.max_r)
        self.history.append((now, desired))
        while self.history and self.history[0][0] < now - self.down_window:
            self.history.popleft()
        if desired > current:
            limit = current + max(math.floor(current * self.up_pct / 100.0), self.up_pods)
            return min(desired, limit, self.max_r)
        if desired < current:
            stabilised = max(d for _, d in self.history)          # rolling maximum over the window
            target = min(max(stabilised, desired), current)
            limit = current - math.floor(current * self.down_pct / 100.0)
            return max(target, limit, lo)
        return current


# ------------------------------------------------------------------ 2. workload
def make_arrivals(segments, rng):
    """segments: [(t0, t1, rate_per_s)]. Poisson arrivals within each (independent of system state = OPEN loop)."""
    out = []
    for t0, t1, rate in segments:
        if rate <= 0:
            continue
        t = t0
        while True:
            t += rng.exponential(1.0 / rate)
            if t >= t1:
                break
            out.append(t)
    return np.array(out)


def lognormal_service(n, mean, sigma, rng):
    mu = math.log(mean) - sigma * sigma / 2
    return rng.lognormal(mu, sigma, n)


# ------------------------------------------------------------------ 3. the simulator
def simulate(arrivals, svc, horizon, slots=4, cold=60.0, scaler=None, static=None, init_replicas=None, dt=0.1,
             metric="outstanding", closed_clients=None, think=1.0, rng=None, trace=None, svc_prior=3.0, mtrace=None):
    """Replay. Open loop by default. Each replica serves up to `slots` requests concurrently; a new replica becomes
    ready `cold` seconds after it is created (billed from creation). FIFO queue. `metric`: 'outstanding' = waiting +
    running requests (what the scaler divides by target); 'queue' = waiting only.
    Returns per-request wait (censored at the horizon for unstarted requests), billing and event counts."""
    n = len(arrivals) if closed_clients is None else None
    reps = []                                   # dicts: ready_at, running (list of finish times), draining
    created = 0

    def spawn(now, count):
        nonlocal created
        for _ in range(count):
            reps.append(dict(ready_at=now + cold, running=[], draining=False))
            created += 1

    cur0 = static if static is not None else (init_replicas if init_replicas is not None else (0 if scaler and scaler.min_r == 0 else max(1, scaler.min_r if scaler else 1)))
    for _ in range(cur0):
        reps.append(dict(ready_at=0.0, running=[], draining=False))

    queue = deque()                             # request ids waiting
    waits, started_at, finished = {}, {}, {}
    arr = list(arrivals) if closed_clients is None else []
    ai = 0
    nxt_id = 0
    client_next = [rng.random() * think for _ in range(closed_clients)] if closed_clients else []
    arr_time = {}
    svc_list = svc
    gpu_s = 0.0
    events = []                                 # (t, new_count)
    t = 0.0
    next_poll = 0.0
    next_hpa = scaler.poll if scaler is not None else 0.0     # the HPA first acts after KEDA's first metric sample
    if scaler is not None:
        scaler.history.append((0.0, len(reps)))                  # previous desired state = the starting replica count
    max_out = 0
    busy_slot_s = 0.0
    n_started_dur, sum_started_dur = 0, 0.0
    arr_np = np.asarray(arrivals) if closed_clients is None else None
    steps = int(horizon / dt)
    for step in range(steps):
        now = step * dt
        # arrivals
        if closed_clients is None:
            while ai < len(arr) and arr[ai] <= now:
                arr_time[ai] = arr[ai]
                queue.append(ai)
                ai += 1
        else:
            for c in range(closed_clients):
                if client_next[c] is not None and client_next[c] <= now:
                    rid = nxt_id; nxt_id += 1
                    arr_time[rid] = client_next[c]
                    queue.append(rid)
                    client_next[c] = None
                    arr.append((rid, c))
        # completions
        for r in reps:
            done = [f for f in r["running"] if f <= now]
            if done:
                r["running"] = [f for f in r["running"] if f > now]
        # remove drained replicas
        reps[:] = [r for r in reps if not (r["draining"] and not r["running"])]
        # starts
        for r in reps:
            if r["draining"] or r["ready_at"] > now:
                continue
            while queue and len(r["running"]) < slots:
                rid = queue.popleft()
                dur = float(svc_list[rid % len(svc_list)]) if closed_clients else float(svc_list[rid])
                waits[rid] = now - arr_time[rid]
                started_at[rid] = now
                r["running"].append(now + dur)
                finished[rid] = now + dur
                busy_slot_s += dur
                n_started_dur += 1; sum_started_dur += dur
        if closed_clients is not None:
            for rid, c in [(rid, c) for rid, c in arr if rid in finished and finished[rid] <= now and client_next[c] is None and (rid, c) in arr]:
                client_next[c] = now + think
                arr.remove((rid, c))
        # scaler
        outstanding = len(queue) + sum(len(r["running"]) for r in reps)
        max_out = max(max_out, outstanding)
        if trace is not None and step % int(15 / dt) == 0:
            trace.append((now, outstanding, len(queue), sum(1 for r in reps if r['ready_at'] <= now and not r['draining']), len(reps)))
        if scaler is not None:
            if metric == "outstanding":
                m = outstanding
            elif metric == "queue":
                m = len(queue)
            else:                                   # 'rate': Little's-law estimate of steady demand, ignores backlog
                lam = (ai - int(np.searchsorted(arr_np, now - 30.0))) / 30.0
                mean_s = sum_started_dur / n_started_dur if n_started_dur else svc_prior
                m = lam * mean_s
            live = [r for r in reps if not r["draining"]]
            cur = len(live)
            if now >= next_poll:
                next_poll += scaler.poll
                new = scaler.keda_poll(now, m, cur)
                if new is not None and new != cur:
                    cur = _resize(reps, cur, new, now, spawn)
                    events.append((now, cur))
            if now >= next_hpa:
                next_hpa += scaler.hpa_period
                if mtrace is not None:
                    mtrace.append((now, outstanding, m))
                if cur >= 1:
                    if not scaler.cached_metrics:
                        scaler.sample = m                  # fresh metric at HPA sync (useCachedMetrics: false)
                    new = scaler.hpa(now, cur)
                    if new != cur:
                        cur = _resize(reps, cur, new, now, spawn)
                        events.append((now, cur))
        gpu_s += len(reps) * dt
    # censor requests that never started
    total = (ai if closed_clients is None else nxt_id)
    w = np.array([waits.get(i, horizon - arr_time[i]) for i in range(total)])
    started = np.array([i in waits for i in range(total)])
    return dict(wait=w, started=started, gpu_s=gpu_s, steps=steps, dt=dt, events=events, created=created, n=total, max_out=max_out,
                busy_slot_s=busy_slot_s, finished=finished, started_at=started_at, arr_time=arr_time, reps_end=len(reps))


def _resize(reps, cur, new, now, spawn):
    if new > cur:
        spawn(now, new - cur)
    else:
        # drain: pick replicas with the least work first, not-yet-ready ones first of all (cancel them)
        live = [r for r in reps if not r["draining"]]
        live.sort(key=lambda r: (r["ready_at"] <= now, len(r["running"])))
        for r in live[: cur - new]:
            if r["ready_at"] > now:
                reps.remove(r)
            else:
                r["draining"] = True
    return new


# ------------------------------------------------------------------ 4. metrics
def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def summarize(res, slo_wait, price_gpu_hr, window=None, arrivals=None):
    w, n = res["wait"], res["n"]
    started = res["started"]
    if window is not None and arrivals is not None:
        sel = (arrivals[:n] >= window[0]) & (arrivals[:n] < window[1])
        w = w[sel]
        started = started[sel]
    # An unstarted request has only a lower-bound wait, never an observed SLO success.
    ok = float((started & (w <= slo_wait)).mean()) if len(w) else float("nan")
    cost = res["gpu_s"] / 3600.0 * price_gpu_hr
    return dict(n=len(w), p50=pct(w, 50), p99=pct(w, 99), attain=ok, gpu_s=res["gpu_s"], cost=cost,
                cost_per_1k=1000.0 * cost / max(1, res["n"]), never_started=int((~res["started"]).sum()), events=len(res["events"]))


# ------------------------------------------------------------------ 5. tests
def selftests():
    res = []

    def check(name, cond):
        res.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # ---- scaler: pure logic
    s = Scaler(target=3.0, min_replicas=1)
    s.sample = 30.0
    check("HPA formula: metric 30, target 3 -> desired ceil(30/3) = 10 (capped by the scale-up policy from 4 replicas: 4 + max(4,4) = 8)",
          s.hpa(0.0, 4) == 8)
    s = Scaler(target=3.0, min_replicas=1); s.sample = 500.0
    seq, cur = [], 1
    for k in range(6):
        cur = s.hpa(15.0 * k, cur); seq.append(cur)
    check(f"scale-up path from 1 replica toward desired 167 under 'max(100%, +4 pods)' per 15 s, clamped at max_replicas=100: {seq}", seq == [5, 10, 20, 40, 80, 100])
    s = Scaler(target=3.0, min_replicas=1); s.sample = 3.3
    check("tolerance: metric/current/target = 1.1 is within 10% -> no change", s.hpa(0.0, 1) == 1)
    s = Scaler(target=3.0, min_replicas=1); s.sample = 3.31
    check("just outside tolerance (ratio 1.1033) -> scales to ceil(3.31/3) = 2", s.hpa(0.0, 1) == 2)

    s = Scaler(target=3.0, min_replicas=1, down_window=300.0); cur = 10
    s.sample = 30.0; cur = s.hpa(0.0, cur)
    s.sample = 3.0; outs = []
    for k in range(1, 22):
        cur = s.hpa(15.0 * k, cur); outs.append(cur)
    check(f"scale-down stabilization: after load drops, replicas stay at 10 for the 300 s window then fall (first fall at step {next(i for i, v in enumerate(outs) if v < 10) + 1} of 21)",
          outs[0] == 10 and outs[-1] < 10 and all(v == 10 for v in outs[:19]))
    s = Scaler(target=3.0, min_replicas=1, down_window=0.0); s.sample = 30.0; cur = s.hpa(0.0, 10)
    s.sample = 3.0
    check("with no stabilization window the same drop scales down at once (to the 100%/period limit, floor 1)", s.hpa(15.0, cur) == 1)

    s = Scaler(target=3.0, min_replicas=0, cooldown=300.0, poll=30.0)
    check("KEDA activation: 0 replicas and metric > 0 -> 1", s.keda_poll(0.0, 5.0, 0) == 1)
    out = [s.keda_poll(30.0 * k, 0.0, 1) for k in range(1, 14)]
    check("KEDA scale to zero only after the metric has been 0 for the cooldown (300 s): returns 0 at the 11th idle poll, not before",
          out[:10] == [None] * 10 and out[10] == 0)
    s2 = Scaler(target=3.0, min_replicas=1)
    check("min_replicas = 1 never scales to zero", all(s2.keda_poll(30.0 * k, 0.0, 1) is None for k in range(1, 100)))
    s3 = Scaler(target=3.0, min_replicas=0, cooldown=300.0)
    for k in range(1, 8):
        s3.keda_poll(30.0 * k, 0.0, 1)
    s3.keda_poll(240.0, 1.0, 1)
    check("a request arriving during the cooldown resets it", s3.idle_since is None)

    # ---- simulator
    rng = np.random.default_rng(0)
    arr = make_arrivals([(0, 600, 4.0)], rng)
    svc = lognormal_service(len(arr), 3.0, 0.5, rng)
    r = simulate(arr, svc, 700.0, slots=4, cold=30.0, static=6)
    check(f"static N: GPU-seconds = N * horizon exactly ({r['gpu_s']:.1f} = 6 * 700)", abs(r["gpu_s"] - 6 * 700.0) < 1e-6)
    check("every request that started did so at or after its arrival (no negative wait)", (r["wait"] >= -1e-9).all())
    comp = sum(1 for i, f in r["finished"].items() if f <= 700.0)
    check(f"with ample capacity (6 replicas for 4 req/s) every one of the {r['n']} arrivals starts and finishes before the horizon ({int(r['started'].sum())} started, {comp} finished)",
          int(r["started"].sum()) == r["n"] and comp == r["n"])
    # slots never exceeded: replay concurrency from start/finish pairs
    pts = sorted([(r["started_at"][i], 1) for i in r["started_at"]] + [(r["finished"][i], -1) for i in r["started_at"]])
    cmax = cc = 0
    for _, d in pts:
        cc += d; cmax = max(cmax, cc)
    check(f"concurrency never exceeds replicas x slots: max in flight {cmax} <= 6 x 4 = 24", cmax <= 24)
    check("busy slot-seconds equal the sum of service durations of the started requests",
          abs(r["busy_slot_s"] - sum(svc[i] for i in r["started_at"])) < 1e-6)

    # Little's law, steady state, plenty of capacity
    rng = np.random.default_rng(1)
    arr = make_arrivals([(0, 3000, 3.0)], rng)
    svc = lognormal_service(len(arr), 3.0, 0.5, rng)
    r = simulate(arr, svc, 3200.0, slots=4, cold=1.0, static=4)
    ids = [i for i in r["finished"] if 300 < r["arr_time"][i] < 2800]
    soj = np.array([r["finished"][i] - r["arr_time"][i] for i in ids])
    wt = np.array([r["wait"][i] for i in ids])
    sv = np.array([svc[i] for i in ids])
    check(f"bookkeeping identity: sojourn = wait + service for every request (max abs error {np.abs(soj - wt - sv).max():.2e})", np.abs(soj - wt - sv).max() < 1e-9)
    check(f"low utilisation (3 req/s offered vs 5.33 capacity): mean wait {wt.mean():.2f} s is below one tick-plus-queueing bound of 1 s; mean sojourn {soj.mean():.2f} s is mean service 3 s plus that wait",
          wt.mean() < 1.0 and abs(soj.mean() - 3.0 - wt.mean()) < 0.15)

    # cold start lower bound from zero
    sc = Scaler(target=3.0, min_replicas=0)
    arr = np.array([10.0]); svc = np.array([2.0])
    r = simulate(arr, svc, 400.0, slots=4, cold=90.0, scaler=sc)
    check(f"scale from zero: first request waits at least the cold start (wait {r['wait'][0]:.1f} s >= 90) plus KEDA's polling lag",
          r["wait"][0] >= 90.0 and r["wait"][0] <= 90.0 + 30.0 + 1.0)

    # billing through scaling paths: scripted scaler, no arrivals
    class Scripted:
        def __init__(self, sched):
            self.sched, self.poll, self.hpa_period, self.min_r, self.cached_metrics, self.history, self.sample = sched, 30.0, 15.0, 1, True, deque(), 0.0
        def keda_poll(self, now, metric, current):
            return None
        def hpa(self, now, current):
            want = current
            for t_from, n in self.sched:
                if now >= t_from:
                    want = n
            return want
    none = np.array([]); nosvc = np.array([])
    r = simulate(none, nosvc, 600.0, slots=4, cold=30.0, scaler=Scripted([(105.0, 5), (300.0, 2)]), init_replicas=2)
    first_up = r["events"][0][0]; first_down = r["events"][1][0]
    exp = 2 * 600.0 + 3 * (first_down - first_up)
    check(f"billing across a scale-up and a drain: events at {first_up:.0f} s (to 5) and {first_down:.0f} s (to 2): GPU-s {r['gpu_s']:.1f} = 2*600 + 3*({first_down:.0f}-{first_up:.0f}) = {exp:.1f} within one tick of drain", abs(r["gpu_s"] - exp) <= 3 * 0.1 * 2 + 1e-6)
    r = simulate(none, nosvc, 600.0, slots=4, cold=30.0, scaler=Scripted([(105.0, 5), (120.0, 2)]), init_replicas=2)
    up, dn = r["events"][0][0], r["events"][1][0]
    exp = 2 * 600.0 + 3 * (dn - up)
    check(f"billing when pods still warming are CANCELLED by a scale-down: up at {up:.0f} s, down at {dn:.0f} s: GPU-s {r['gpu_s']:.1f} = 2*600 + 3*({dn:.0f}-{up:.0f}) = {exp:.1f}; replicas alive at the end {r['reps_end']} == 2",
          abs(r["gpu_s"] - exp) <= 3 * 0.1 + 1e-6 and r["reps_end"] == 2)

    # fresh versus cached metric at the HPA
    rng = np.random.default_rng(11)
    a = make_arrivals([(31, 200, 10.0)], rng); v = lognormal_service(len(a), 3.0, 0.5, rng)
    firsts = []
    for cached in (False, True):
        r = simulate(a, v, 200.0, slots=4, cold=30.0, scaler=Scaler(target=3.0, min_replicas=1, cached_metrics=cached), init_replicas=1)
        firsts.append(r["events"][0][0])
    check(f"a spike starts at t=31 just after KEDA's poll at t=30: with the default FRESH metric the HPA first scales at t={firsts[0]:.0f} (its t=45 sync); with useCachedMetrics it waits for the next KEDA poll and scales at t={firsts[1]:.0f}",
          abs(firsts[0] - 45.0) <= 0.2 and abs(firsts[1] - 60.0) <= 0.2)

    # rate metric (Little's law) tracks outstanding in steady state
    rng = np.random.default_rng(12)
    a = make_arrivals([(0, 1500, 3.0)], rng); v = lognormal_service(len(a), 3.0, 0.5, rng)
    mt = []
    simulate(a, v, 1500.0, slots=4, cold=1.0, scaler=Scaler(target=3.0, min_replicas=4, max_replicas=4), init_replicas=4, metric="rate", mtrace=mt)
    sel = [(o, m) for t_, o, m in mt if 300 <= t_ <= 1400]
    mo, mm = np.mean([x[0] for x in sel]), np.mean([x[1] for x in sel])
    check(f"rate metric (arrivals in the last 30 s / 30 x mean service) tracks the true outstanding count in steady state: mean {mm:.2f} vs {mo:.2f} (within 10%; both near 3 req/s x 3 s = 9)", abs(mm - mo) / mo < 0.10)

    # determinism
    def run_once():
        rng = np.random.default_rng(7)
        a = make_arrivals([(0, 300, 5.0), (300, 400, 20.0)], rng)
        v = lognormal_service(len(a), 3.0, 0.5, rng)
        return simulate(a, v, 500.0, cold=45.0, scaler=Scaler(target=3.0, min_replicas=1))
    a, b = run_once(), run_once()
    check("same seed, same result (waits, GPU-seconds, scaling events)", np.array_equal(a["wait"], b["wait"]) and a["gpu_s"] == b["gpu_s"] and a["events"] == b["events"])
    return all(res)


def experiments():
    PRICE = 4.0           # ASSUMED $ per GPU-hour (one GPU per replica; see the chapter for tensor parallelism)
    SLO = 5.0             # wait <= 5 s
    SL, MEAN = 4, 3.0     # 4 concurrent slots per replica, mean service 3 s => capacity 4/3 = 1.33 req/s per replica

    def workload(segs, seed):
        rng = np.random.default_rng(seed)
        a = make_arrivals(segs, rng)
        return a, lognormal_service(len(a), MEAN, 0.5, rng)

    def agg(segs, seeds, horizon, make_kw, window=None, cold=60.0, init=None):
        """Run the same policy over several seeds; return dict of lists."""
        out = dict(gpu=[], att=[], attw=[], p99w=[], peak=[], cost1k=[], events=[], created=[])
        for sd in seeds:
            a, v = workload(segs, sd)
            kw = make_kw()
            r = simulate(a, v, horizon, slots=SL, cold=cold, init_replicas=init, **kw)
            s = summarize(r, SLO, PRICE)
            out["gpu"].append(s["gpu_s"]); out["att"].append(s["attain"]); out["p99w"].append(s["p99"]); out["cost1k"].append(s["cost_per_1k"])
            out["events"].append(s["events"]); out["created"].append(r["created"])
            out["peak"].append(kw["static"] if "static" in kw else max([c for _, c in r["events"]] + [init or 0]))
            if window is not None:
                out["attw"].append(summarize(r, SLO, PRICE, window=window, arrivals=a)["attain"])
        return out

    def fm(x, fmt="{:.0f}"):
        x = np.asarray(x, float)
        return (fmt + " [" + fmt + "-" + fmt + "]").format(x.mean(), x.min(), x.max())

    print(f"\nAssumptions: {SL} slots/replica, mean service {MEAN}s (capacity {SL / MEAN:.2f} req/s per replica), SLO wait <= {SLO}s, ${PRICE}/GPU-hour (ASSUMED)")
    print("Scaler: HPA period 15 s (fresh metric by default, i.e. useCachedMetrics=false), KEDA poll 30 s (0<->1 only unless cached), tolerance 0.1,")
    print("        scale-up max(100%,+4)/15 s, scale-down window 300 s, cooldown 300 s, max 100 replicas; target 3 outstanding per replica")
    print("Cells show mean [min-max] over seeds. One global FIFO queue (real pods hold their own queues): see the chapter's limits.")

    print("\n== 1. Steady diurnal-like load, 1 hour (2 -> 6 -> 10 -> 4 req/s), cold start 60 s; 5 seeds ==")
    segs = [(0, 900, 2.0), (900, 1800, 6.0), (1800, 2700, 10.0), (2700, 3600, 4.0)]
    seeds = (1, 2, 3, 4, 5)
    print(f"  {'policy':36s} {'GPU-s':>22s} {'attainment':>20s} {'peak replicas':>16s} {'$/1k req':>20s}")
    pol = [("static 4 (sized for the average)", lambda: dict(static=4), None),
           ("static 9 (sized for the peak)", lambda: dict(static=9), None),
           ("autoscale default (fresh metric)", lambda: dict(scaler=Scaler(target=3.0, min_replicas=1)), 2),
           ("autoscale, useCachedMetrics (30 s)", lambda: dict(scaler=Scaler(target=3.0, min_replicas=1, cached_metrics=True)), 2),
           ("autoscale, scale-up capped at +8", lambda: dict(scaler=Scaler(target=3.0, min_replicas=1, up_percent=0.0, up_pods=8)), 2),
           ("autoscale, rate (Little's-law) metric", lambda: dict(scaler=Scaler(target=3.0, min_replicas=1), metric="rate"), 2),
           ("autoscale, rate metric + 60 s window", lambda: dict(scaler=Scaler(target=3.0, min_replicas=1, down_window=60.0), metric="rate"), 2)]
    for name, mk, init in pol:
        o = agg(segs, seeds, 3700.0, mk, init=init)
        print(f"  {name:36s} {fm(o['gpu']):>22s} {fm(o['att'], '{:.3f}'):>20s} {fm(o['peak']):>16s} {fm(o['cost1k'], '{:.2f}'):>20s}")

    print("\n== 2. A 10x burst (2 -> 20 req/s for 120 s at t=600); attainment in the window [600, 1020); 3 seeds ==")
    segs = [(0, 600, 2.0), (600, 720, 20.0), (720, 1500, 2.0)]
    for maxr in (100, 24):
        print(f"  -- maxReplicaCount = {maxr}")
        print(f"  {'cold':>5s} {'floor':>6s} {'attain (burst window)':>24s} {'p99 wait (window)':>20s} {'peak replicas':>15s} {'GPU-s':>20s}")
        for cold in (15.0, 60.0, 180.0):
            for fl in (2, 8, 16):
                o = agg(segs, (2, 3, 4), 1600.0, lambda: dict(scaler=Scaler(target=3.0, min_replicas=fl, max_replicas=maxr)), window=(600, 1020), cold=cold, init=max(fl, 2))
                print(f"  {cold:5.0f} {fl:6d} {fm(o['attw'], '{:.3f}'):>24s} {fm(o['p99w'], '{:.1f}'):>20s} {fm(o['peak']):>15s} {fm(o['gpu']):>20s}")
    print("  -- floor 16, maxReplicaCount 24, cold 60 s: how sharp is the sizing? (capacity at 16 replicas = 21.3 req/s)")
    print(f"  {'burst req/s':>12s} {'attain (burst window)':>24s} {'p99 wait (window)':>20s} {'peak replicas':>15s}")
    for rate in (18.0, 20.0, 22.0, 25.0):
        sg = [(0, 600, 2.0), (600, 720, rate), (720, 1500, 2.0)]
        o = agg(sg, (2, 3, 4), 1600.0, lambda: dict(scaler=Scaler(target=3.0, min_replicas=16, max_replicas=24)), window=(600, 1020), cold=60.0, init=16)
        print(f"  {rate:12.0f} {fm(o['attw'], '{:.3f}'):>24s} {fm(o['p99w'], '{:.1f}'):>20s} {fm(o['peak']):>15s}")

    print("\n== 3. Scale to zero: 3 req/s for 300 s, idle, then 1 req/s for 60 s. Cold start 60 s. 3 seeds; first-request wait on seed 3 ==")
    print("   (Requests are assumed to queue somewhere while there are zero replicas: see the chapter for why that needs a gateway.)")
    print(f"  {'policy':34s} {'GPU-s':>20s} {'attain (all)':>20s} {'replicas created':>18s}")
    for start in (1500.0, 1515.0):
        segs = [(0, 300, 3.0), (start, start + 60, 1.0)]
        print(f"  -- wake-up burst starts at t={start:.0f} s ({'exactly on a 30 s KEDA poll boundary' if start % 30 == 0 else '15 s after a poll boundary'})")
        pol = [("min=0, default HPA behaviour", lambda: dict(scaler=Scaler(target=3.0, min_replicas=0))),
               ("min=0, no scale-down window", lambda: dict(scaler=Scaler(target=3.0, min_replicas=0, down_window=0.0))),
               ("min=0, scale-up capped at +1", lambda: dict(scaler=Scaler(target=3.0, min_replicas=0, up_percent=0.0, up_pods=1))),
               ("min=0, no window, +1 cap", lambda: dict(scaler=Scaler(target=3.0, min_replicas=0, down_window=0.0, up_percent=0.0, up_pods=1))),
               ("min=0, rate metric", lambda: dict(scaler=Scaler(target=3.0, min_replicas=0), metric="rate")),
               ("min=1, default HPA behaviour", lambda: dict(scaler=Scaler(target=3.0, min_replicas=1)))]
        for name, mk in pol:
            o = agg(segs, (1, 2, 3), 2200.0, mk, init=3)
            a, v = workload(segs, 3)
            r = simulate(a, v, 2200.0, slots=SL, cold=60.0, init_replicas=3, **mk())
            idx = int(np.searchsorted(a, start))
            print(f"  {name:34s} {fm(o['gpu']):>20s} {fm(o['att'], '{:.3f}'):>20s} {fm(o['created']):>18s}   first wait {r['wait'][idx]:6.1f} s")

    print("\n== 4. Testing the scaler: open-loop burst vs closed-loop clients (same 120 s window; the closed loops are NOT offered the burst) ==")
    segs = [(0, 300, 2.0), (300, 420, 20.0), (420, 900, 2.0)]
    a, v = workload(segs, 4)
    ro = simulate(a, v, 1000.0, slots=SL, cold=60.0, scaler=Scaler(target=3.0, min_replicas=1), init_replicas=2)
    so = summarize(ro, SLO, PRICE)
    print(f"  open loop (Poisson, 20 req/s burst): peak outstanding {ro['max_out']:5d}  p99 wait {so['p99']:7.1f} s  attain {so['attain']:.3f}  peak replicas {max(c for _, c in ro['events'])}")
    for clients, think in ((8, 3.0), (64, 0.5)):
        rc = simulate(None, v, 1000.0, slots=SL, cold=60.0, scaler=Scaler(target=3.0, min_replicas=1), init_replicas=2, closed_clients=clients, think=think, rng=np.random.default_rng(0))
        sc = summarize(rc, SLO, PRICE)
        pk = max([c for _, c in rc["events"]] + [2])
        print(f"  closed loop, {clients:2d} clients, think {think} s: peak outstanding {rc['max_out']:5d}  p99 wait {sc['p99']:7.1f} s  attain {sc['attain']:.3f}  peak replicas {pk}")
    print("  A closed loop can only ever have `clients` requests outstanding. It exercises the scaler only if clients >= the demand you want to test, and its arrival rate falls as latency rises.")

    print("\n== 5. Flapping: load alternating 2 <-> 12 req/s every 150 s, by scale-down stabilization window; 3 seeds ==")
    segs = []
    for k in range(12):
        segs.append((150 * k, 150 * (k + 1), 12.0 if k % 2 else 2.0))
    print(f"  {'down window':>11s} {'scale events':>14s} {'replicas created':>18s} {'warm-up GPU-s (created x 60)':>30s} {'attainment':>18s} {'GPU-s':>20s}")
    for win in (0.0, 60.0, 300.0, 600.0):
        o = agg(segs, (1, 2, 3), 1850.0, lambda: dict(scaler=Scaler(target=3.0, min_replicas=1, down_window=win)), init=2)
        print(f"  {win:11.0f} {fm(o['events']):>14s} {fm(o['created']):>18s} {fm(np.array(o['created']) * 60.0):>30s} {fm(o['att'], '{:.3f}'):>18s} {fm(o['gpu']):>20s}")

    print("\n== 6. Metric choice, steady 8 req/s for 30 minutes starting from 8 replicas; 3 seeds ==")
    print(f"  {'metric':14s} {'attainment':>18s} {'p99 wait':>18s} {'GPU-s':>22s} {'peak replicas':>15s}")
    for m in ("outstanding", "queue", "rate"):
        o = agg([(0, 1800, 8.0)], (1, 2, 3), 1900.0, lambda: dict(scaler=Scaler(target=3.0, min_replicas=1), metric=m), init=8)
        print(f"  {m:14s} {fm(o['att'], '{:.3f}'):>18s} {fm(o['p99w'], '{:.1f}'):>18s} {fm(o['gpu']):>22s} {fm(o['peak']):>15s}")

    print("\n== 7. Where does the overshoot come from? Ablations on a 2 -> 12 req/s step (held 600 s); 5 seeds. Steady need at 12 req/s is about 12 replicas ==")
    segs = [(0, 300, 2.0), (300, 900, 12.0), (900, 1500, 2.0)]
    seeds = (1, 2, 3, 4, 5)
    print(f"  {'variant':44s} {'peak replicas':>16s} {'attain (step window)':>22s} {'GPU-s':>22s}")
    ab = [("baseline: cold 60 s, window 300 s, default policy", 60.0, lambda: dict(scaler=Scaler(target=3.0, min_replicas=1))),
          ("cold start 1 s (backlog cannot build)", 1.0, lambda: dict(scaler=Scaler(target=3.0, min_replicas=1))),
          ("scale-down window 0 (nothing is held)", 60.0, lambda: dict(scaler=Scaler(target=3.0, min_replicas=1, down_window=0.0))),
          ("rate metric (ignores backlog)", 60.0, lambda: dict(scaler=Scaler(target=3.0, min_replicas=1), metric="rate")),
          ("scale-up capped at +8", 60.0, lambda: dict(scaler=Scaler(target=3.0, min_replicas=1, up_percent=0.0, up_pods=8))),
          ("useCachedMetrics (30 s)", 60.0, lambda: dict(scaler=Scaler(target=3.0, min_replicas=1, cached_metrics=True)))]
    for name, cold, mk in ab:
        o = agg(segs, seeds, 1600.0, mk, window=(300, 900), cold=cold, init=2)
        print(f"  {name:44s} {fm(o['peak']):>16s} {fm(o['attw'], '{:.3f}'):>22s} {fm(o['gpu']):>22s}")


if __name__ == "__main__":
    t0 = time.time()
    print("self-tests:")
    ok = selftests()
    if "--selftest" not in sys.argv:
        experiments()
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}   ({time.time() - t0:.0f} s)")
    sys.exit(0 if ok else 1)
