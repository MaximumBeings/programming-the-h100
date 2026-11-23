"""Chapter 36: a rigorous serving-benchmark harness, tested against mock OpenAI-compatible streaming servers.

Run:   python3 ch36_benchmark_teardown.py            # self-tests + a mock three-engine teardown (~1-2 min, CPU)
       python3 ch36_benchmark_teardown.py --selftest # self-tests only

What is real here: the client (open-loop Poisson load, intended-time latency, streaming timestamps, failure
accounting), the statistics (percentiles, bootstrap CIs), the Pareto/knee logic, the SSE and chunked parsers, and
the run manifest. What is NOT real: the three "engines" are mock servers with invented parameters. No vLLM, SGLang
or TensorRT-LLM result is produced or implied by this file.
"""
import asyncio
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict


# =============================================================== 1. streaming parsers
class SSEParser:
    """Incremental Server-Sent-Events parser. Feed arbitrary byte chunks; get complete 'data:' payloads back."""
    def __init__(self):
        self.buf = b""

    def feed(self, chunk: bytes):
        self.buf += chunk
        out = []
        while True:
            i = self.buf.find(b"\n\n")
            if i < 0:
                break
            event, self.buf = self.buf[:i], self.buf[i + 2:]
            for line in event.split(b"\n"):
                line = line.rstrip(b"\r")
                if line.startswith(b"data:"):
                    out.append(line[5:].strip().decode())
        return out


async def read_body_chunks(reader, headers):
    """Yield body bytes as they arrive, handling both 'Transfer-Encoding: chunked' and close-delimited bodies."""
    if headers.get("transfer-encoding", "").lower() == "chunked":
        while True:
            size_line = await reader.readline()
            if not size_line:
                return
            size = int(size_line.strip().split(b";")[0], 16)
            if size == 0:
                await reader.readline()
                return
            data = await reader.readexactly(size)
            await reader.readexactly(2)                      # trailing CRLF
            yield data
    else:
        while True:
            data = await reader.read(4096)
            if not data:
                return
            yield data


# =============================================================== 2. mock engine (stands in for a real server)
@dataclass
class EngineProfile:
    name: str
    base_ms: float            # per-token base latency
    per_seq_ms: float         # extra per-token latency per concurrently running sequence
    capacity: int             # max concurrently running sequences (others queue)
    prefill_ms_per_100tok: float = 4.0
    error_rate: float = 0.0
    gpus: int = 1
    chunked: bool = True


class MockEngine:
    def __init__(self, prof: EngineProfile, seed=0):
        self.p, self.rng = prof, random.Random(seed)
        self.sem = asyncio.Semaphore(prof.capacity)
        self.active = 0
        self.server = None
        self.port = None

    async def start(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self):
        self.server.close()
        await self.server.wait_closed()

    @staticmethod
    def _chunk(data: bytes) -> bytes:
        return f"{len(data):x}\r\n".encode() + data + b"\r\n"

    async def handle(self, reader, writer):
        try:
            await reader.readline()                                   # request line
            clen = 0
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b""):
                    break
                if line.lower().startswith(b"content-length:"):
                    clen = int(line.split(b":")[1])
            body = json.loads(await reader.readexactly(clen))
            n_out, n_in = int(body["max_tokens"]), int(body.get("prompt_tokens", 128))
            if self.rng.random() < self.p.error_rate:
                writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return
            hdr = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            hdr += b"Transfer-Encoding: chunked\r\n\r\n" if self.p.chunked else b"Connection: close\r\n\r\n"
            writer.write(hdr)
            put = self._chunk if self.p.chunked else (lambda b: b)
            async with self.sem:                                      # queueing happens here
                self.active += 1
                try:
                    await asyncio.sleep(n_in / 100 * self.p.prefill_ms_per_100tok / 1e3)
                    for _ in range(n_out):
                        await asyncio.sleep((self.p.base_ms + self.p.per_seq_ms * self.active) / 1e3)
                        writer.write(put(b'data: {"choices":[{"text":"x"}]}\n\n'))
                        await writer.drain()
                finally:
                    self.active -= 1
            writer.write(put(b'data: {"usage":{"completion_tokens":%d}}\n\ndata: [DONE]\n\n' % n_out))
            if self.p.chunked:
                writer.write(b"0\r\n\r\n")
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()


# =============================================================== 3. client: open-loop and closed-loop load
@dataclass
class Rec:
    intended: float                       # when the schedule said to send
    sent: float = math.nan                # when we actually sent
    first: float = math.nan               # first streamed token
    end: float = math.nan
    ntok: int = 0
    ok: bool = False
    err: str = ""
    itl: list = field(default_factory=list)


async def one_request(port, n_in, n_out, rec: Rec, timeout=30.0):
    try:
        async def go():
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            payload = json.dumps({"max_tokens": n_out, "prompt_tokens": n_in, "stream": True}).encode()
            rec.sent = time.perf_counter()
            writer.write(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                         b"Content-Length: %d\r\n\r\n" % len(payload) + payload)
            await writer.drain()
            status = (await reader.readline()).decode()
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b""):
                    break
                k, _, v = line.decode().partition(":")
                headers[k.strip().lower()] = v.strip()
            if " 200 " not in status:
                rec.err = status.strip()
                writer.close()
                return
            parser, last, done = SSEParser(), None, False
            async for chunk in read_body_chunks(reader, headers):
                for ev in parser.feed(chunk):
                    now = time.perf_counter()
                    if ev == "[DONE]":
                        done = True
                    elif '"text"' in ev:
                        if math.isnan(rec.first):
                            rec.first = now
                        elif last is not None:
                            rec.itl.append(now - last)
                        last = now
                        rec.ntok += 1
            rec.end = time.perf_counter()
            rec.ok = done and rec.ntok == n_out                       # a truncated stream is a FAILURE
            if not rec.ok:
                rec.err = f"incomplete stream: {rec.ntok}/{n_out} tokens, done={done}"
            writer.close()
        await asyncio.wait_for(go(), timeout)
    except asyncio.TimeoutError:
        rec.err = "timeout"
    except Exception as e:                                           # noqa: BLE001 - benchmark must not crash
        rec.err = f"{type(e).__name__}: {e}"
    return rec


async def run_open_loop(port, rate, n_req, seed, n_in_med=256, n_out_med=48, stall_at=None, stall_s=0.0):
    """Poisson arrivals scheduled in advance. Each request is launched at its INTENDED time regardless of how
    earlier requests are doing. stall_at/stall_s inject a client-side stall (to demonstrate coordinated omission)."""
    rng = random.Random(seed)
    t, sched = 0.0, []
    for _ in range(n_req):
        t += rng.expovariate(rate)
        sched.append((t, max(8, int(rng.lognormvariate(math.log(n_in_med), .5))),
                      max(4, int(rng.lognormvariate(math.log(n_out_med), .4)))))
    t0 = time.perf_counter() + 0.05
    recs, tasks = [], []
    for i, (at, ni, no) in enumerate(sched):
        rec = Rec(intended=t0 + at)
        recs.append(rec)
        delay = rec.intended - time.perf_counter()
        if stall_at is not None and i == stall_at:
            time.sleep(stall_s)                                      # blocks the whole client event loop
            delay = rec.intended - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one_request(port, ni, no, rec)))
    await asyncio.gather(*tasks)
    return recs


async def run_closed_loop(port, concurrency, n_req, seed, n_in_med=256, n_out_med=48):
    """N workers; each sends its next request only after the previous one finishes. (Self-throttling.)"""
    rng = random.Random(seed)
    jobs = [(max(8, int(rng.lognormvariate(math.log(n_in_med), .5))),
             max(4, int(rng.lognormvariate(math.log(n_out_med), .4)))) for _ in range(n_req)]
    recs, it = [], iter(jobs)

    async def worker():
        for ni, no in it:
            r = Rec(intended=time.perf_counter())                    # "intended" == now: that is the omission
            recs.append(r)
            await one_request(port, ni, no, r)
    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return recs


# =============================================================== 4. metrics and statistics
def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return math.nan
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(recs, warmup=10, slo_ttft=0.5, slo_tpot=0.05, from_intended=True, usd_per_gpu_hour=None, gpus=1):
    recs = sorted(recs, key=lambda r: r.intended)[warmup:]                   # discard warm-up by arrival order
    ok = [r for r in recs if r.ok]
    base = (lambda r: r.intended) if from_intended else (lambda r: r.sent)
    ttft = [r.first - base(r) for r in ok]
    e2e = [r.end - base(r) for r in ok]
    tpot = [(r.end - r.first) / max(1, r.ntok - 1) for r in ok]
    wall = max(r.end for r in ok) - min(r.intended for r in recs) if ok else math.nan
    toks = sum(r.ntok for r in ok)
    good = sum(1 for a, b in zip(ttft, tpot) if a <= slo_ttft and b <= slo_tpot)
    out = dict(n=len(recs), failures=len(recs) - len(ok), success_rate=len(ok) / len(recs),
               ttft_p50=percentile(ttft, .5), ttft_p99=percentile(ttft, .99),
               tpot_p50=percentile(tpot, .5), tpot_p99=percentile(tpot, .99),
               e2e_p99=percentile(e2e, .99), tok_s=toks / wall,
               goodput_rps=good / wall, slo_attain=good / len(recs))      # denominator = ALL requests, failures included
    if usd_per_gpu_hour:
        out["usd_per_mtok"] = usd_per_gpu_hour * gpus * wall / 3600 / (toks / 1e6)
    return out


def bootstrap_ci(xs, stat, B=800, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    vals = sorted(stat([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return vals[int(B * alpha / 2)], vals[int(B * (1 - alpha / 2)) - 1]


def diff_ci(a, b, stat, B=800, alpha=0.05, seed=0):
    """CI for stat(a) - stat(b). If it excludes 0, the engines differ beyond resampling noise (for THIS workload)."""
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        vals.append(stat(ra) - stat(rb))
    vals.sort()
    return vals[int(B * alpha / 2)], vals[int(B * (1 - alpha / 2)) - 1]


def pareto(points, minimize, maximize=()):
    """Indices of non-dominated points. A point is dominated if another is no worse on every axis and better on one."""
    def key(p):
        return [p[m] for m in minimize] + [-p[m] for m in maximize]
    keys = [key(p) for p in points]
    keep = []
    for i, ki in enumerate(keys):
        dominated = any(all(kj[t] <= ki[t] for t in range(len(ki))) and any(kj[t] < ki[t] for t in range(len(ki)))
                        for j, kj in enumerate(keys) if j != i)
        if not dominated:
            keep.append(i)
    return keep


def knee(rows, min_attain=0.99):
    """Highest offered rate whose SLO attainment (failures counted) stays at or above min_attain."""
    good = [r for r in rows if r["slo_attain"] >= min_attain]
    return max((r["rate"] for r in good), default=None)


# =============================================================== 5. run manifest (reproducibility)
def manifest(extra=None):
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
        except Exception:                                            # noqa: BLE001
            return None
    m = dict(python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine(),
             argv=sys.argv, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             nvidia_smi=sh(["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem", "--format=csv,noheader"])
             if shutil.which("nvidia-smi") else None)
    m.update(extra or {})
    m["config_hash"] = hashlib.sha256(json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return m


# =============================================================== 6. self-tests
def selftests():
    results = []

    def check(name, cond):
        results.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # SSE parser: result independent of how the byte stream is cut
    stream = b'data: {"a":1}\n\ndata: {"b":2}\r\n\r\n: comment\n\ndata: [DONE]\n\n'
    whole = SSEParser().feed(stream)
    rng = random.Random(1)
    same = True
    for _ in range(200):
        p, out, i = SSEParser(), [], 0
        while i < len(stream):
            j = i + rng.randint(1, 7)
            out += p.feed(stream[i:j]); i = j
        same &= (out == whole)
    check("SSE parser identical under 200 random chunkings", same and whole == ['{"a":1}', '{"b":2}', "[DONE]"])

    # percentile vs numpy
    try:
        import numpy as np
        xs = [rng.random() for _ in range(101)]
        check("percentile matches numpy (p50,p99)", all(abs(percentile(xs, q) - float(np.percentile(xs, q * 100))) < 1e-12
                                                         for q in (.5, .99)))
    except ImportError:
        check("percentile matches numpy (skipped: numpy missing)", True)

    # Pareto: frontier is non-dominated, everything else is dominated
    pts = [dict(c=rng.random(), l=rng.random()) for _ in range(200)]
    kept = pareto(pts, ["c", "l"])
    nd = all(not any(pts[j]["c"] <= pts[i]["c"] and pts[j]["l"] <= pts[i]["l"] and
                     (pts[j]["c"] < pts[i]["c"] or pts[j]["l"] < pts[i]["l"]) for j in range(200)) for i in kept)
    dom = all(any(pts[j]["c"] <= pts[i]["c"] and pts[j]["l"] <= pts[i]["l"] and
                  (pts[j]["c"] < pts[i]["c"] or pts[j]["l"] < pts[i]["l"]) for j in range(200))
              for i in range(200) if i not in kept)
    check(f"Pareto frontier correct on 200 random points ({len(kept)} kept)", nd and dom)

    # bootstrap CI coverage: the 95% CI of a median should contain the true median ~95% of the time
    cover, trials = 0, 150
    for s in range(trials):
        r = random.Random(100 + s)
        sample = [r.gauss(0, 1) for _ in range(120)]
        lo, hi = bootstrap_ci(sample, lambda v: percentile(v, .5), B=200, seed=s)
        cover += lo <= 0.0 <= hi
    check(f"bootstrap 95% CI coverage of true median = {cover/trials:.2f} (want about 0.90-0.99)", 0.88 <= cover / trials <= 0.995)

    # summarize: failures count against SLO attainment, warm-up is discarded
    recs = []
    for i in range(40):
        r = Rec(intended=float(i), sent=float(i), first=i + 0.1, end=i + 0.5, ntok=11, ok=(i % 10 != 0))
        recs.append(r)
    s = summarize(recs, warmup=0, slo_ttft=0.5, slo_tpot=0.05)
    check("summarize counts failures (4 of 40) and attainment uses ALL requests", s["failures"] == 4 and abs(s["slo_attain"] - 36 / 40) < 1e-9)
    s2 = summarize(recs, warmup=10, slo_ttft=0.5, slo_tpot=0.05)
    check("warm-up discards the first 10 requests by arrival", s2["n"] == 30)
    # Pareto with a maximize axis and a knee
    rows = [dict(rate=r, slo_attain=a) for r, a in ((10, 1.0), (20, 0.995), (30, 0.7), (40, 0.4))]
    check("knee = highest rate with attainment >= 0.99", knee(rows) == 20)
    return all(results)


# =============================================================== 7. coordinated-omission demonstration
async def demo_coordinated_omission():
    prof = EngineProfile("narrow", base_ms=6, per_seq_ms=0.5, capacity=4)
    eng = MockEngine(prof); await eng.start()
    open_recs = await run_open_loop(eng.port, rate=20, n_req=80, seed=5)
    closed_recs = await run_closed_loop(eng.port, concurrency=4, n_req=80, seed=5)
    await eng.stop()
    so, sc = summarize(open_recs, warmup=5), summarize(closed_recs, warmup=5)
    print(f"  open-loop  @20 rps offered: ttft_p99={so['ttft_p99']*1e3:8.1f} ms   e2e_p99={so['e2e_p99']*1e3:8.1f} ms")
    print(f"  closed-loop, 4 workers    : ttft_p99={sc['ttft_p99']*1e3:8.1f} ms   e2e_p99={sc['e2e_p99']*1e3:8.1f} ms")
    # client-side stall: measuring from SEND time hides the stall; from INTENDED time exposes it
    eng2 = MockEngine(EngineProfile("wide", base_ms=4, per_seq_ms=0.1, capacity=64)); await eng2.start()
    recs = await run_open_loop(eng2.port, rate=40, n_req=100, seed=6, stall_at=40, stall_s=0.6)
    await eng2.stop()
    recs = sorted(recs, key=lambda r: r.intended)[5:]
    ok = [r for r in recs if r.ok]
    share_send = sum(1 for r in ok if r.first - r.sent > 0.3) / len(ok)
    share_int = sum(1 for r in ok if r.first - r.intended > 0.3) / len(ok)
    a = summarize(recs, warmup=0, from_intended=False)
    b = summarize(recs, warmup=0, from_intended=True)
    print(f"  0.6 s client stall at request 40 (of 100, 40 req/s offered):")
    print(f"    share of requests with TTFT > 300 ms, measured from SEND time     = {share_send:5.1%}  (only requests already in flight)")
    print(f"    share of requests with TTFT > 300 ms, measured from INTENDED time = {share_int:5.1%}  (also those that should have been sent during the stall)")
    print(f"    mean TTFT: from send {sum(r.first - r.sent for r in ok)/len(ok)*1e3:6.1f} ms   from intended {sum(r.first - r.intended for r in ok)/len(ok)*1e3:6.1f} ms")
    return so, sc, a, b


# =============================================================== 8. mock three-engine teardown
async def teardown(rates=(15, 35, 70), repeats=2, n_req=90):
    profiles = [EngineProfile("engine-A", 4.0, 0.15, 16, gpus=2),
                EngineProfile("engine-B", 3.0, 0.35, 32, gpus=2),
                EngineProfile("engine-C", 5.0, 0.08, 8, gpus=1)]
    USD = 4.0
    table = {p.name: [] for p in profiles}
    pooled = {p.name: {} for p in profiles}
    for p in profiles:
        for rate in rates:
            runs = []
            for rep in range(repeats):
                eng = MockEngine(p, seed=rep); await eng.start()
                recs = await run_open_loop(eng.port, rate, n_req, seed=1000 * rep + rate)
                await eng.stop()
                runs.append(recs)
            allr = [r for rs in runs for r in rs]
            row = summarize(allr, warmup=10 * repeats, usd_per_gpu_hour=USD, gpus=p.gpus)
            row.update(rate=rate, run_p99=[summarize(rs, warmup=10)["ttft_p99"] for rs in runs])
            table[p.name].append(row)
            pooled[p.name][rate] = [r.first - r.intended for r in sorted(allr, key=lambda r: r.intended)[10 * repeats:] if r.ok]
    return profiles, table, pooled


def print_teardown(profiles, table, pooled):
    print("\n  engine     rate  ttft_p99(ms)  [95% CI]            tpot_p99(ms)  tok/s  fail  attain  $/Mtok  run-to-run p99 (ms)")
    for p in profiles:
        for row in table[p.name]:
            lo, hi = bootstrap_ci(pooled[p.name][row["rate"]], lambda v: percentile(v, .99), B=300)
            rr = " / ".join(f"{x*1e3:.0f}" for x in row["run_p99"])
            print(f"  {p.name}  {row['rate']:4d}  {row['ttft_p99']*1e3:11.1f}  [{lo*1e3:6.1f},{hi*1e3:7.1f}]  {row['tpot_p99']*1e3:11.1f}  "
                  f"{row['tok_s']:5.0f}  {row['failures']:4d}  {row['slo_attain']:6.2f}  {row['usd_per_mtok']:6.2f}  {rr}")
    print("\n  knee (highest offered rate with >=99% SLO attainment, failures counted):")
    for p in profiles:
        print(f"    {p.name}: {knee(table[p.name])} req/s")
    # significance for one comparison at the middle rate
    mid = sorted({r['rate'] for r in table['engine-A']})[1]
    lo, hi = diff_ci(pooled['engine-A'][mid], pooled['engine-B'][mid], lambda v: percentile(v, .99))
    verdict = "engines differ beyond resampling noise" if lo > 0 or hi < 0 else "NOT distinguishable at this sample size"
    print(f"\n  p99 TTFT difference A-B at {mid} req/s: 95% CI [{lo*1e3:.1f}, {hi*1e3:.1f}] ms -> {verdict}")
    # Pareto over (cost, p99 TTFT) with throughput maximized, all (engine, rate) points
    pts, labels = [], []
    for p in profiles:
        for row in table[p.name]:
            if row["slo_attain"] >= 0.99:                       # only points that actually meet the SLO are candidates
                pts.append(dict(cost=row["usd_per_mtok"], ttft=row["ttft_p99"], tput=row["tok_s"]))
                labels.append(f"{p.name}@{row['rate']}")
    kept = pareto(pts, ["cost", "ttft"], ["tput"])
    print("  SLO-feasible Pareto-optimal (cost, p99 TTFT, throughput):", ", ".join(labels[i] for i in kept))
    print("  excluded as infeasible (SLO attainment < 99%):",
          ", ".join(f"{p.name}@{r['rate']}" for p in profiles for r in table[p.name] if r["slo_attain"] < 0.99))


def main():
    print("== self-tests ==")
    ok = selftests()
    if "--selftest" in sys.argv:
        sys.exit(0 if ok else 1)
    print("\n== coordinated omission demonstration (mock servers) ==")
    asyncio.run(demo_coordinated_omission())
    print("\n== mock three-engine teardown (engines are INVENTED; shows the method, not any real result) ==")
    profiles, table, pooled = asyncio.run(teardown())
    print_teardown(profiles, table, pooled)
    print("\n== run manifest ==")
    print(json.dumps(manifest(dict(note="mock run")), indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
