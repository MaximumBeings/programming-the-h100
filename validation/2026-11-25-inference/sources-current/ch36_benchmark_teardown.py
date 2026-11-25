"""Chapter 36: a rigorous serving-benchmark harness, tested against mock OpenAI-compatible streaming servers.

Run:   python3 ch36_benchmark_teardown.py            # self-tests + demos + mock three-engine teardown (a few minutes, CPU)
       python3 ch36_benchmark_teardown.py --selftest # self-tests only (about 20 s)
       python3 ch36_benchmark_teardown.py run --url http://127.0.0.1:8000/v1/completions --model NAME \
               --rate 10 --n 600 --warmup 50 --seed 1 --out run.jsonl     # drive a REAL OpenAI-compatible server

What is real here: the client (open-loop Poisson load, intended-time latency, streaming timestamps, failure
accounting), the statistics (percentiles, bootstrap CIs), the Pareto/knee logic, the SSE and chunked parsers, and
the run manifest. What is NOT real: the three "engines" are mock servers with invented parameters. No vLLM, SGLang
or TensorRT-LLM result is produced or implied by this file.
"""
import argparse
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
from urllib.parse import urlsplit
from dataclasses import dataclass, field, asdict


# =============================================================== 1. streaming parsers
class SSEParser:
    """Incremental Server-Sent-Events parser. Feed arbitrary byte chunks; get complete event payloads back.

    Handles LF, CRLF and lone-CR line endings (including a CR/LF pair split across two chunks), ignores comment
    lines, and joins multi-line 'data:' fields with '\n' as the SSE spec says. An event is complete only when its
    terminating blank line has arrived; a trailing partial event is never emitted (the client then sees no [DONE]
    and counts the request as failed)."""
    def __init__(self):
        self.buf = b""

    def feed(self, chunk: bytes):
        data = self.buf + chunk
        hold = b""
        if data.endswith(b"\r"):                       # may be the first half of CRLF: keep it for the next feed
            data, hold = data[:-1], b"\r"
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        out = []
        while True:
            i = data.find(b"\n\n")
            if i < 0:
                break
            event, data = data[:i], data[i + 2:]
            fields = []
            for line in event.split(b"\n"):
                if line.startswith(b"data:"):
                    v = line[5:]
                    fields.append((v[1:] if v.startswith(b" ") else v).decode())
            if fields:
                out.append("\n".join(fields))
        self.buf = data + hold
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
    merge: int = 1             # tokens per SSE event (real engines sometimes batch several)
    truncate_prob: float = 0.0 # probability the stream is cut short without [DONE]


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
            n_out = int(body["max_tokens"])
            n_in = len(body["prompt"].split()) if "prompt" in body else int(body.get("prompt_tokens", 128))
            if self.rng.random() < self.p.error_rate:
                writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return
            hdr = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            hdr += b"Transfer-Encoding: chunked\r\n\r\n" if self.p.chunked else b"Connection: close\r\n\r\n"
            writer.write(hdr)
            put = self._chunk if self.p.chunked else (lambda b: b)
            cut = self.rng.random() < self.p.truncate_prob
            n_send = n_out // 2 if cut else n_out
            async with self.sem:                                      # queueing happens here
                self.active += 1
                try:
                    await asyncio.sleep(n_in / 100 * self.p.prefill_ms_per_100tok / 1e3)
                    sent = 0
                    while sent < n_send:
                        k = min(self.p.merge, n_send - sent)
                        await asyncio.sleep(k * (self.p.base_ms + self.p.per_seq_ms * self.active) / 1e3)
                        writer.write(put(b'data: {"choices":[{"text":"' + b"x" * k + b'"}]}\n\n'))
                        await writer.drain()
                        sent += k
                finally:
                    self.active -= 1
            if cut:
                return                                                # connection closed mid-stream, no [DONE]
            writer.write(put(b'data: {"choices":[],"usage":{"completion_tokens":%d}}\n\ndata: [DONE]\n\n' % n_out))
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
    first: float = math.nan               # first streamed token event
    end: float = math.nan
    ntok: int = 0                         # output tokens: server-reported usage if present, else content events
    nevents: int = 0                      # content events seen (may be < ntok if the server merges tokens)
    ok: bool = False
    err: str = ""
    itl: list = field(default_factory=list)


@dataclass
class Target:
    host: str = "127.0.0.1"
    port: int = 8000
    path: str = "/v1/completions"
    model: str = "mock"
    ignore_eos: bool = True               # force exact output length so every engine does equal work

    @staticmethod
    def from_url(url, model="mock", ignore_eos=True):
        u = urlsplit(url)
        if u.scheme != "http":
            raise ValueError("this client speaks plain http; put a local TLS-terminating proxy in front of https servers")
        return Target(u.hostname, u.port or 80, u.path or "/v1/completions", model, ignore_eos)


def make_payload(t: Target, n_in, n_out):
    p = {"model": t.model, "prompt": " ".join(["hello"] * n_in), "max_tokens": n_out, "temperature": 0,
         "stream": True, "stream_options": {"include_usage": True}}
    if t.ignore_eos:
        p["ignore_eos"] = True                                        # accepted by vLLM and SGLang; check your engine
    return json.dumps(p).encode()


async def one_request(target, n_in, n_out, rec: Rec, timeout=30.0, payload=None):
    if isinstance(target, int):                                       # backwards compatible: bare port
        target = Target(port=target)
    try:
        async def go():
            reader, writer = await asyncio.open_connection(target.host, target.port)
            body = payload or make_payload(target, n_in, n_out)
            rec.sent = time.perf_counter()
            writer.write(f"POST {target.path} HTTP/1.1\r\nHost: {target.host}:{target.port}\r\n"
                         f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
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
                rec.err = status.strip() or "empty status line"
                writer.close()
                return
            parser, last, done, usage = SSEParser(), None, False, None
            async for chunk in read_body_chunks(reader, headers):
                for ev in parser.feed(chunk):
                    now = time.perf_counter()
                    if ev.strip() == "[DONE]":
                        done = True
                        continue
                    try:
                        obj = json.loads(ev)
                    except ValueError:
                        rec.err = "unparseable event"
                        continue
                    if isinstance(obj, dict) and obj.get("usage"):
                        usage = obj["usage"].get("completion_tokens", usage)
                    text = ""
                    for ch in (obj.get("choices") or []) if isinstance(obj, dict) else []:
                        text += ch.get("text") or (ch.get("delta") or {}).get("content") or ""
                    if text:                                          # one content-bearing event
                        if math.isnan(rec.first):
                            rec.first = now
                        elif last is not None:
                            rec.itl.append(now - last)
                        last = now
                        rec.nevents += 1
            rec.end = time.perf_counter()
            rec.ntok = usage if usage is not None else rec.nevents    # prefer the server's own token count
            rec.ok = (done and not rec.err and rec.ntok == n_out and
                      math.isfinite(rec.first) and rec.end >= rec.first)                       # truncated stream or wrong length = FAILURE
            if not rec.ok:
                rec.err = f"incomplete or wrong-length stream: {rec.ntok}/{n_out} tokens, done={done}"
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


def _collect(recs, warmup, slo_ttft, slo_tpot, from_intended):
    """Per-RUN statistics. Warm-up is discarded per run, by arrival order, so every run loses its own cold start."""
    recs = sorted(recs, key=lambda r: r.intended)[warmup:]
    ok = [r for r in recs if r.ok]
    base = (lambda r: r.intended) if from_intended else (lambda r: r.sent)
    ttft = [r.first - base(r) for r in ok]
    tpot = [(r.end - r.first) / (r.ntok - 1) if r.ntok > 1 else 0.0 for r in ok]
    return dict(n=len(recs), ok=len(ok), ttft=ttft, tpot=tpot, e2e=[r.end - base(r) for r in ok],
                wall=(max(r.end for r in ok) - min(r.intended for r in recs)) if ok else 0.0,
                toks=sum(r.ntok for r in ok),
                good=sum(1 for a, b in zip(ttft, tpot) if a <= slo_ttft and b <= slo_tpot))


def _finalize(parts, usd_per_gpu_hour, gpus):
    n = sum(p["n"] for p in parts); nok = sum(p["ok"] for p in parts)
    ttft = [x for p in parts for x in p["ttft"]]; tpot = [x for p in parts for x in p["tpot"]]
    e2e = [x for p in parts for x in p["e2e"]]
    wall = sum(p["wall"] for p in parts)                      # sum of per-run windows: the gap BETWEEN runs is excluded
    toks, good = sum(p["toks"] for p in parts), sum(p["good"] for p in parts)
    out = dict(n=n, failures=n - nok, success_rate=nok / n if n else math.nan,
               ttft_p50=percentile(ttft, .5), ttft_p99=percentile(ttft, .99),
               tpot_p50=percentile(tpot, .5), tpot_p99=percentile(tpot, .99), e2e_p99=percentile(e2e, .99),
               tok_s=toks / wall if wall else 0.0, goodput_rps=good / wall if wall else 0.0,
               slo_attain=good / n if n else math.nan,          # denominator = ALL requests, failures included
               ttft_samples=ttft)
    if usd_per_gpu_hour and toks:
        out["usd_per_mtok"] = usd_per_gpu_hour * gpus * wall / 3600 / (toks / 1e6)
    return out


def summarize(recs, warmup=10, slo_ttft=0.5, slo_tpot=0.05, from_intended=True, usd_per_gpu_hour=None, gpus=1):
    return _finalize([_collect(recs, warmup, slo_ttft, slo_tpot, from_intended)], usd_per_gpu_hour, gpus)


def summarize_runs(runs, warmup=10, slo_ttft=0.5, slo_tpot=0.05, from_intended=True, usd_per_gpu_hour=None, gpus=1):
    """Pool several repeats of the same configuration: warm-up per run, throughput over the sum of run windows."""
    return _finalize([_collect(r, warmup, slo_ttft, slo_tpot, from_intended) for r in runs], usd_per_gpu_hour, gpus)


def bootstrap_ci(xs, stat, B=800, alpha=0.05, seed=0):
    """Percentile bootstrap over requests. Good for medians and means; for p99 it under-covers unless n is large
    (about 1000 or more), because a resample can never exceed the sample maximum. See the coverage self-test."""
    rng = random.Random(seed)
    n = len(xs)
    vals = sorted(stat(rng.choices(xs, k=n)) for _ in range(B))
    return vals[int(B * alpha / 2)], vals[int(B * (1 - alpha / 2)) - 1]


def diff_ci(a, b, stat, B=800, alpha=0.05, seed=0):
    """CI for stat(a) - stat(b). If it excludes 0, the engines differ beyond resampling noise (for THIS workload)."""
    rng = random.Random(seed)
    vals = sorted(stat(rng.choices(a, k=len(a))) - stat(rng.choices(b, k=len(b))) for _ in range(B))
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
    smi = None
    if shutil.which("nvidia-smi"):
        # field names differ between drivers; try the richer query first
        for q in ("name,driver_version,clocks.sm,clocks.mem,persistence_mode,power.limit,clocks_throttle_reasons.active",
                  "name,driver_version,clocks.sm,clocks.mem,persistence_mode,power.limit"):
            smi = sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"])
            if smi:
                break
    m = dict(python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine(),
             argv=sys.argv, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), nvidia_smi=smi)
    m.update(extra or {})
    m["config_hash"] = hashlib.sha256(json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return m


# =============================================================== 6. self-tests
def selftests():
    results = []

    def check(name, cond):
        results.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = random.Random(1)
    # SSE parser: result independent of how the byte stream is cut, for LF, CRLF-only and multi-line data
    streams = {
        "LF/CRLF mix": (b'data: {"a":1}\n\ndata: {"b":2}\r\n\r\n: comment\n\ndata: [DONE]\n\n',
                        ['{"a":1}', '{"b":2}', "[DONE]"]),
        "CRLF only": (b'data: {"a":1}\r\n\r\ndata: {"b":2}\r\n\r\ndata: [DONE]\r\n\r\n',
                      ['{"a":1}', '{"b":2}', "[DONE]"]),
        "multi-line data": (b'data: line1\ndata: line2\n\ndata: [DONE]\n\n', ["line1\nline2", "[DONE]"]),
        "partial last event": (b'data: one\n\ndata: tw', ["one"]),
    }
    for name, (stream, expect) in streams.items():
        whole = SSEParser().feed(stream)
        same = True
        for _ in range(200):
            p, out, i = SSEParser(), [], 0
            while i < len(stream):
                j = i + rng.randint(1, 7)
                out += p.feed(stream[i:j]); i = j
            same &= (out == whole)
        check(f"SSE parser, {name}: 200 random chunkings agree and equal {expect!r}", same and whole == expect)

    try:
        import numpy as np
        xs = [rng.random() for _ in range(101)]
        check("percentile matches numpy (p50,p99)", all(abs(percentile(xs, q) - float(np.percentile(xs, q * 100))) < 1e-12
                                                         for q in (.5, .99)))
    except ImportError:
        check("percentile matches numpy (skipped: numpy missing)", True)

    pts = [dict(c=rng.random(), l=rng.random()) for _ in range(200)]
    kept = pareto(pts, ["c", "l"])
    dom_ = lambda j, i: (pts[j]["c"] <= pts[i]["c"] and pts[j]["l"] <= pts[i]["l"] and
                         (pts[j]["c"] < pts[i]["c"] or pts[j]["l"] < pts[i]["l"]))
    nd = all(not any(dom_(j, i) for j in range(200)) for i in kept)
    dm = all(any(dom_(j, i) for j in range(200)) for i in range(200) if i not in kept)
    check(f"Pareto frontier correct on 200 random points ({len(kept)} kept)", nd and dm)

    # bootstrap coverage: median (n=120) and p99 at two sample sizes. The p99 numbers are INFORMATIONAL for n=160.
    def coverage(n, q, truth, trials, B, gen):
        c = 0
        for sd in range(trials):
            r = random.Random(500 + sd)
            smp = [gen(r) for _ in range(n)]
            lo, hi = bootstrap_ci(smp, lambda v: percentile(v, q), B=B, seed=sd)
            c += lo <= truth <= hi
        return c / trials
    sig = 0.6
    ln = lambda r: math.exp(r.gauss(0, sig))
    c_med = coverage(120, .5, 1.0, 150, 200, ln)
    c_160 = coverage(160, .99, math.exp(sig * 2.3263479), 150, 200, ln)
    c_1000 = coverage(1000, .99, math.exp(sig * 2.3263479), 100, 150, ln)
    print(f"      bootstrap coverage of true value (nominal 0.95): median n=120 -> {c_med:.2f}; "
          f"p99 n=160 -> {c_160:.2f}; p99 n=1000 -> {c_1000:.2f}")
    check("bootstrap median coverage within 0.88-0.995", 0.88 <= c_med <= 0.995)
    check("bootstrap p99 coverage at n=1000 is at least 0.85 and not worse than at n=160 (minus noise)",
          c_1000 >= 0.85 and c_1000 >= c_160 - 0.05)

    # summarize: failures count against SLO attainment, warm-up discarded
    recs = [Rec(intended=float(i), sent=float(i), first=i + 0.1, end=i + 0.5, ntok=11, ok=(i % 10 != 0)) for i in range(40)]
    s = summarize(recs, warmup=0)
    check("summarize counts failures (4 of 40); attainment uses ALL requests", s["failures"] == 4 and abs(s["slo_attain"] - 36 / 40) < 1e-9)
    check("warm-up discards the first 10 requests by arrival", summarize(recs, warmup=10)["n"] == 30)

    # pooled runs: warm-up must come out of EACH run, and the gap between runs must not count as wall time
    def mk(offset):
        return [Rec(intended=offset + i * 0.1, sent=offset + i * 0.1, first=offset + i * 0.1 + 0.05,
                    end=offset + i * 0.1 + 0.2, ntok=11, ok=True) for i in range(50)]
    r1, r2 = mk(0.0), mk(1000.0)                                    # run 2 starts 1000 s later
    sp = summarize_runs([r1, r2], warmup=10)
    wall_expected = 2 * ((49 * 0.1 + 0.2) - 10 * 0.1)               # per-run window after dropping 10 warm-ups
    check("pooled summary: 40 per run kept (80 total), warm-up taken from each run", sp["n"] == 80)
    check("pooled throughput uses the sum of run windows, not the 1000 s gap",
          abs(sp["tok_s"] - 80 * 11 / wall_expected) < 1e-6)
    rows = [dict(rate=r, slo_attain=a) for r, a in ((10, 1.0), (20, 0.995), (30, 0.7), (40, 0.4))]
    check("knee = highest rate with attainment >= 0.99", knee(rows) == 20)
    check("make_payload is a valid OpenAI-style streaming request",
          json.loads(make_payload(Target(model="m"), 3, 5)) == {"model": "m", "prompt": "hello hello hello", "max_tokens": 5,
                                                              "temperature": 0, "stream": True,
                                                              "stream_options": {"include_usage": True}, "ignore_eos": True})
    return all(results)


async def async_selftests():
    """Tests that need sockets or streams: chunked and close-delimited bodies, errors, truncation, merged events, timeout."""
    results = []

    def check(name, cond):
        results.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # read_body_chunks on synthetic readers (no sockets)
    r = asyncio.StreamReader()
    r.feed_data(b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"); r.feed_eof()
    got = b"".join([c async for c in read_body_chunks(r, {"transfer-encoding": "chunked"})])
    r2 = asyncio.StreamReader(); r2.feed_data(b"hello world"); r2.feed_eof()
    got2 = b"".join([c async for c in read_body_chunks(r2, {})])
    check("read_body_chunks: chunked body and close-delimited body both decode", got == got2 == b"hello world")

    async def one(prof, n_out=20, timeout=5.0):
        eng = MockEngine(prof); await eng.start()
        rec = Rec(intended=time.perf_counter())
        await one_request(Target(port=eng.port), 16, n_out, rec, timeout=timeout)
        await eng.stop()
        return rec

    base = dict(base_ms=2, per_seq_ms=0, capacity=4)
    r = await one(EngineProfile("ok", **base))
    check("normal request succeeds with 20 tokens", r.ok and r.ntok == 20 and r.nevents == 20)
    r = await one(EngineProfile("nochunk", chunked=False, **base))
    check("close-delimited (non-chunked) body succeeds", r.ok and r.ntok == 20)
    r = await one(EngineProfile("merged", merge=4, **base))
    check("merged events (4 tokens/event): success judged by server usage, not event count",
          r.ok and r.ntok == 20 and r.nevents == 5)
    r = await one(EngineProfile("err", error_rate=1.0, **base))
    check("HTTP 503 is recorded as a failure with its status", (not r.ok) and "503" in r.err)
    r = await one(EngineProfile("trunc", truncate_prob=1.0, **base))
    check("truncated stream (no [DONE]) is a failure", (not r.ok) and "incomplete" in r.err)
    r = await one(EngineProfile("slow", base_ms=50, per_seq_ms=0, capacity=4), n_out=20, timeout=0.2)
    check("client timeout is recorded as a failure", (not r.ok) and r.err == "timeout")

    # error injection flows through to the SLO-attainment denominator
    eng = MockEngine(EngineProfile("flaky", error_rate=0.3, **base), seed=3); await eng.start()
    recs = await run_open_loop(eng.port, rate=50, n_req=60, seed=2, n_out_med=10)
    await eng.stop()
    s = summarize(recs, warmup=0, slo_ttft=5.0, slo_tpot=5.0)
    check(f"with 30% injected errors: {s['failures']}/60 failures counted and attainment = {s['slo_attain']:.2f} < 0.85",
          s["failures"] > 0 and s["slo_attain"] < 0.85)
    return all(results)


# =============================================================== 7. coordinated-omission demonstrations
async def demo_coordinated_omission():
    """All numbers vary run to run (wall-clock asyncio). The assertions test qualitative relations only."""
    prof = EngineProfile("narrow", base_ms=6, per_seq_ms=0.5, capacity=4)
    # capacity ~ 4 slots / (~52 tokens * ~8 ms + prefill) ~ 9 req/s. Below it: stable. Above it: no steady state.
    out = {}
    for label, rate, n in (("below capacity", 5, 90), ("2x overload, short run", 18, 40), ("2x overload, long run", 18, 160)):
        eng = MockEngine(prof); await eng.start()
        out[label] = summarize(await run_open_loop(eng.port, rate=rate, n_req=n, seed=5), warmup=3)
        await eng.stop()
    eng = MockEngine(prof); await eng.start()
    closed = summarize(await run_closed_loop(eng.port, concurrency=4, n_req=90, seed=5), warmup=3)
    await eng.stop()
    print("  open-loop, Poisson arrivals, capacity of the mock server about 9 req/s:")
    for label, s in out.items():
        print(f"    {label:24s}: ttft_p99 = {s['ttft_p99']*1e3:8.1f} ms   e2e_p99 = {s['e2e_p99']*1e3:8.1f} ms   (n={s['n']})")
    print(f"  closed-loop, 4 workers    : ttft_p99 = {closed['ttft_p99']*1e3:8.1f} ms   e2e_p99 = {closed['e2e_p99']*1e3:8.1f} ms   (n={closed['n']})")
    # client-side stall: measuring from SEND time hides the stall; from INTENDED time exposes it
    eng2 = MockEngine(EngineProfile("wide", base_ms=4, per_seq_ms=0.1, capacity=64)); await eng2.start()
    recs = await run_open_loop(eng2.port, rate=40, n_req=100, seed=6, stall_at=40, stall_s=0.6)
    await eng2.stop()
    ok = [r for r in sorted(recs, key=lambda r: r.intended)[5:] if r.ok]
    share_send = sum(1 for r in ok if r.first - r.sent > 0.3) / len(ok)
    share_int = sum(1 for r in ok if r.first - r.intended > 0.3) / len(ok)
    m_send = sum(r.first - r.sent for r in ok) / len(ok); m_int = sum(r.first - r.intended for r in ok) / len(ok)
    print("  0.6 s client stall at request 40 (of 100, 40 req/s offered):")
    print(f"    share with TTFT > 300 ms measured from SEND time     = {share_send:5.1%}")
    print(f"    share with TTFT > 300 ms measured from INTENDED time = {share_int:5.1%}")
    print(f"    mean TTFT: from send {m_send*1e3:6.1f} ms   from intended {m_int*1e3:6.1f} ms")
    below, short, long_ = out["below capacity"], out["2x overload, short run"], out["2x overload, long run"]
    checks = {
        "below capacity, open-loop TTFT p99 is within 30x of closed-loop (a real but bounded gap)":
            below["ttft_p99"] < 30 * max(closed["ttft_p99"], 0.02),
        "under overload, open-loop TTFT p99 is at least 5x the closed-loop figure":
            long_["ttft_p99"] > 5 * max(closed["ttft_p99"], 0.02),
        "under overload, p99 keeps growing with run length (no steady state)": long_["ttft_p99"] > short["ttft_p99"],
        "stall: intended-time measurement counts more slow requests than send-time": share_int > share_send,
        "stall: intended-time mean TTFT exceeds send-time mean TTFT": m_int > m_send,
    }
    for name, c in checks.items():
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    return all(checks.values())


# =============================================================== 8. mock three-engine teardown
WARMUP = 10


async def teardown(rates=(15, 35, 70), repeats=3, n_req=150):
    profiles = [EngineProfile("engine-A", 4.0, 0.15, 16, gpus=2),
                EngineProfile("engine-B", 3.0, 0.35, 32, gpus=2),
                EngineProfile("engine-C", 5.0, 0.08, 8, gpus=1)]
    USD = 4.0
    table = {p.name: [] for p in profiles}
    for p in profiles:
        for rate in rates:
            runs = []
            for rep in range(repeats):
                eng = MockEngine(p, seed=rep); await eng.start()
                runs.append(await run_open_loop(eng.port, rate, n_req, seed=1000 * rep + rate))
                await eng.stop()
            row = summarize_runs(runs, warmup=WARMUP, usd_per_gpu_hour=USD, gpus=p.gpus)
            row.update(rate=rate, run_p99=[summarize(rs, warmup=WARMUP)["ttft_p99"] for rs in runs])
            table[p.name].append(row)
    return profiles, table


def print_teardown(profiles, table):
    print("\n  SAMPLE OUTPUT (wall-clock timing: your numbers will differ, and borderline verdicts can flip)")
    print("  engine     rate     n  ttft_p99(ms)  [95% CI]            tpot_p99(ms)  tok/s  fail  attain  $/Mtok  per-run p99 (ms)")
    for p in profiles:
        for row in table[p.name]:
            lo, hi = bootstrap_ci(row["ttft_samples"], lambda v: percentile(v, .99), B=800)
            rr = " / ".join(f"{x*1e3:.0f}" for x in row["run_p99"])
            print(f"  {p.name}  {row['rate']:4d}  {row['n']:4d}  {row['ttft_p99']*1e3:11.1f}  [{lo*1e3:6.1f},{hi*1e3:7.1f}]  {row['tpot_p99']*1e3:11.1f}  "
                  f"{row['tok_s']:5.0f}  {row['failures']:4d}  {row['slo_attain']:6.2f}  {row['usd_per_mtok']:6.2f}  {rr}")
    print("\n  knee (highest offered rate with >=99% SLO attainment, failures counted):")
    for p in profiles:
        print(f"    {p.name}: {knee(table[p.name])} req/s")
    mid = sorted({r['rate'] for r in table['engine-A']})[1]
    a = [r for r in table['engine-A'] if r['rate'] == mid][0]["ttft_samples"]
    b = [r for r in table['engine-B'] if r['rate'] == mid][0]["ttft_samples"]
    lo, hi = diff_ci(a, b, lambda v: percentile(v, .99))
    verdict = "engines differ beyond resampling noise" if lo > 0 or hi < 0 else "NOT distinguishable at this sample size"
    print(f"\n  p99 TTFT difference A-B at {mid} req/s: 95% CI [{lo*1e3:.1f}, {hi*1e3:.1f}] ms -> {verdict}")
    pts, labels = [], []
    for p in profiles:
        for row in table[p.name]:
            if row["slo_attain"] >= 0.99:
                pts.append(dict(cost=row["usd_per_mtok"], ttft=row["ttft_p99"], tput=row["tok_s"]))
                labels.append(f"{p.name}@{row['rate']}")
    kept = pareto(pts, ["cost", "ttft"], ["tput"])
    print("  SLO-feasible Pareto-optimal (cost, p99 TTFT, throughput):", ", ".join(labels[i] for i in kept))
    print("  excluded as infeasible (SLO attainment < 99%):",
          ", ".join(f"{p.name}@{r['rate']}" for p in profiles for r in table[p.name] if r["slo_attain"] < 0.99))
    cheapest = min((r["usd_per_mtok"], p.name, r["rate"]) for p in profiles for r in table[p.name])
    print(f"  note: the lowest $/Mtok anywhere is {cheapest[1]}@{cheapest[2]} ({cheapest[0]:.2f}); "
          f"it is {'feasible' if cheapest[1] + '@' + str(cheapest[2]) in labels else 'NOT SLO-feasible'}")


# =============================================================== 9. command-line client for real servers
async def run_real(args):
    target = Target.from_url(args.url, args.model, ignore_eos=not args.no_ignore_eos)
    recs = await run_open_loop(target, args.rate, args.n, args.seed, n_in_med=args.n_in, n_out_med=args.n_out)
    s = summarize(recs, warmup=args.warmup, slo_ttft=args.slo_ttft, slo_tpot=args.slo_tpot,
                  usd_per_gpu_hour=args.usd_per_gpu_hour, gpus=args.gpus)
    s.pop("ttft_samples")
    print(json.dumps(s, indent=2))
    errs = {}
    for r in recs:
        if not r.ok:
            errs[r.err.split(":")[0]] = errs.get(r.err.split(":")[0], 0) + 1
    print("failure reasons:", errs or "none")
    if args.out:
        with open(args.out, "w") as f:
            for r in recs:
                f.write(json.dumps(asdict(r)) + "\n")
        print(f"raw per-request records written to {args.out}")
    print(json.dumps(manifest(dict(url=args.url, model=args.model, rate=args.rate, n=args.n, seed=args.seed)), indent=2))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        ap = argparse.ArgumentParser(prog="ch36_benchmark_teardown.py run",
                                     description="Open-loop load against a real OpenAI-compatible /v1/completions server")
        ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
        ap.add_argument("--rate", type=float, required=True, help="offered requests per second (Poisson)")
        ap.add_argument("--n", type=int, default=600); ap.add_argument("--warmup", type=int, default=50)
        ap.add_argument("--seed", type=int, default=1)
        ap.add_argument("--n-in", type=int, default=256, help="median prompt length in words (about tokens)")
        ap.add_argument("--n-out", type=int, default=128, help="median output tokens (lognormal spread)")
        ap.add_argument("--slo-ttft", type=float, default=0.5); ap.add_argument("--slo-tpot", type=float, default=0.05)
        ap.add_argument("--usd-per-gpu-hour", type=float, default=None); ap.add_argument("--gpus", type=int, default=1)
        ap.add_argument("--no-ignore-eos", action="store_true", help="do not send ignore_eos (outputs may stop early)")
        ap.add_argument("--out", default=None)
        asyncio.run(run_real(ap.parse_args(sys.argv[2:])))
        return
    print("== self-tests ==")
    ok = selftests()
    print("== self-tests needing sockets (mock server on localhost) ==")
    ok &= asyncio.run(async_selftests())
    if "--selftest" in sys.argv:
        sys.exit(0 if ok else 1)
    print("\n== coordinated omission demonstrations (mock servers; SAMPLE OUTPUT, varies per run) ==")
    ok &= asyncio.run(demo_coordinated_omission())
    print("\n== mock three-engine teardown (engines are INVENTED; shows the method, not any real result) ==")
    profiles, table = asyncio.run(teardown())
    print_teardown(profiles, table)
    print("\n== run manifest ==")
    print(json.dumps(manifest(dict(note="mock run")), indent=2))
    print("\nALL PASS" if ok else "\nFAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
