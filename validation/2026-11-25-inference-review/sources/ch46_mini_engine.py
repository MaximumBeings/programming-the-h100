#!/usr/bin/env python3
"""Chapter 46 - Capstone: a mini inference engine that joins Chapters 43, 44 and 45.

A tiny real transformer (NumPy, float64) is served by an engine with a paged KV cache, a
continuous-batching scheduler with preemption and chunked prefill, and a block-level prefix
cache. Every output is checked against a naive reference that recomputes everything.

Run:  python3 ch46_mini_engine.py          # self-tests + experiments
      python3 ch46_mini_engine.py tests    # self-tests only
Needs ch43_paged_kv_cache.py, ch44_continuous_batching.py, ch45_radix_prefix_cache.py alongside.
"""
import math
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch43_paged_kv_cache import BlockAllocator            # noqa: E402
import ch44_continuous_batching as c44                    # noqa: E402
import ch45_radix_prefix_cache as c45                     # noqa: E402

T0, CP, CD = c44.T0, c44.CP, c44.CD


# --------------------------------------------------------------------------
# A tiny decoder-only transformer (random weights; the point is the machinery)
# --------------------------------------------------------------------------
class TinyLM:
    def __init__(self, V=96, d=32, H=2, L=2, ff=64, max_pos=1024, seed=0):
        if any(not isinstance(x, (int, np.integer)) or x <= 0 for x in (V, d, H, L, ff, max_pos)):
            raise ValueError("model dimensions must be positive integers")
        if d % H or d % 2:
            raise ValueError("model width must be even and divisible by the head count")
        r = np.random.default_rng(seed)
        self.V, self.d, self.H, self.L, self.D = V, d, H, L, d // H
        sc = lambda n, m: r.standard_normal((n, m)) / math.sqrt(n)
        self.E = r.standard_normal((V, d)) * 0.7
        self.layers = [dict(Wq=sc(d, d), Wk=sc(d, d), Wv=sc(d, d), Wo=sc(d, d), W1=sc(d, ff), W2=sc(ff, d))
                       for _ in range(L)]
        pos = np.arange(max_pos)[:, None]
        i = np.arange(d // 2)[None, :]
        ang = pos / (10000 ** (2 * i / d))
        self.P = np.concatenate([np.sin(ang), np.cos(ang)], axis=1)

    @staticmethod
    def rms(x):
        return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)

    def embed(self, tokens, start):
        return self.E[tokens] + self.P[start:start + len(tokens)]

    def qkv(self, l, x):
        h = self.rms(x)
        c = len(x)
        W = self.layers[l]
        return ((h @ W["Wq"]).reshape(c, self.H, self.D), (h @ W["Wk"]).reshape(c, self.H, self.D),
                (h @ W["Wv"]).reshape(c, self.H, self.D))

    def finish_layer(self, l, x, attn):
        W = self.layers[l]
        x = x + attn.reshape(len(x), -1) @ W["Wo"]
        h = self.rms(x)
        return x + np.maximum(h @ W["W1"], 0) @ W["W2"]

    def logits(self, x):
        return self.rms(x) @ self.E.T

    # ---- reference: dense causal attention over the whole sequence, no cache ----
    def forward_full(self, tokens):
        x = self.embed(np.asarray(tokens), 0)
        T = len(x)
        mask = np.tril(np.ones((T, T), bool))
        for l in range(self.L):
            q, k, v = self.qkv(l, x)
            s = np.einsum("qhd,khd->hqk", q, k) / math.sqrt(self.D)
            s = np.where(mask[None], s, -np.inf)
            s -= s.max(-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(-1, keepdims=True)
            x = self.finish_layer(l, x, np.einsum("hqk,khd->qhd", p, v))
        return self.logits(x)

    def ref_generate(self, prompt, n_new):
        toks = list(prompt)
        out = []
        for _ in range(n_new):
            nxt = int(np.argmax(self.forward_full(toks)[-1]))
            toks.append(nxt)
            out.append(nxt)
        return out


# --------------------------------------------------------------------------
# KV pool for all layers + attention kernels over block tables
# --------------------------------------------------------------------------
class KVPool:
    def __init__(self, model, n_blocks, B):
        self.B = B
        self.K = np.zeros((model.L, n_blocks, B, model.H, model.D))
        self.V = np.zeros_like(self.K)


def attn_gather_chunk(q, pool, l, table, start):
    """Chunk prefill: gather the sequence's K/V, dense causal attention for c queries."""
    c, H, D = q.shape
    end = start + c
    Kall = pool.K[l, table].reshape(-1, H, D)[:end]
    Vall = pool.V[l, table].reshape(-1, H, D)[:end]
    s = np.einsum("qhd,khd->hqk", q, Kall) / math.sqrt(D)
    mask = np.arange(end)[None, :] <= (start + np.arange(c))[:, None]
    s = np.where(mask[None], s, -np.inf)
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", p, Vall)


def attn_paged_decode(q, pool, l, table, length):
    """Decode: one query walks the block table with an online softmax; no gather."""
    H, D = q.shape
    m = np.full(H, -np.inf)
    s_ = np.zeros(H)
    acc = np.zeros((H, D))
    remaining = length
    for b in table:
        n = min(pool.B, remaining)
        if n <= 0:
            break
        remaining -= n
        sc = np.einsum("hd,thd->ht", q, pool.K[l, b, :n]) / math.sqrt(D)
        mn = np.maximum(m, sc.max(1))
        scale = np.exp(m - mn)
        p = np.exp(sc - mn[:, None])
        s_ = s_ * scale + p.sum(1)
        acc = acc * scale[:, None] + np.einsum("ht,thd->hd", p, pool.V[l, b, :n])
        m = mn
    return acc / s_[:, None]


def forward_chunk(model, pool, table, tokens, start):
    """Run `tokens` at positions start.. through the model, writing K/V into the sequence's blocks.
    Returns the logits of the last token."""
    B = pool.B
    x = model.embed(np.asarray(tokens), start)
    c = len(tokens)
    pos = start + np.arange(c)
    blk = np.asarray(table)[pos // B]
    off = pos % B
    for l in range(model.L):
        q, k, v = model.qkv(l, x)
        pool.K[l, blk, off] = k
        pool.V[l, blk, off] = v
        if c == 1:
            a = attn_paged_decode(q[0], pool, l, table, start + 1)[None]
        else:
            a = attn_gather_chunk(q, pool, l, table, start)
        x = model.finish_layer(l, x, a)
    return model.logits(x[-1:])[0]


# --------------------------------------------------------------------------
# Block-level prefix cache (exact keys: parent node + the block's tokens)
# --------------------------------------------------------------------------
class PrefixCache:
    class N:
        __slots__ = ("id", "block", "parent", "last", "children")

    def __init__(self, B, alloc):
        self.B, self.alloc = B, alloc
        self.nodes = {}           # (parent_id, block tokens) -> N
        self.by_id = {0: None}
        self.clock = 0
        self._id = 0

    def match(self, toks, max_blocks):
        self.clock += 1
        pid, out = 0, []
        for i in range(max_blocks):
            n = self.nodes.get((pid, tuple(toks[i * self.B:(i + 1) * self.B])))
            if n is None:
                break
            n.last = self.clock
            out.append(n.block)
            pid = n.id
        return out

    def register(self, toks, table):
        self.clock += 1
        pid = 0
        for i in range(len(toks) // self.B):
            key = (pid, tuple(toks[i * self.B:(i + 1) * self.B]))
            n = self.nodes.get(key)
            if n is None:
                n = PrefixCache.N()
                self._id += 1
                n.id, n.block, n.parent, n.last, n.children = self._id, table[i], pid, self.clock, 0
                self.nodes[key] = n
                self.by_id[n.id] = n
                if pid:
                    self.by_id[pid].children += 1
                self.alloc.incref(n.block)
            else:
                n.last = self.clock
            pid = n.id

    def evict_one(self):
        best = None
        for key, n in self.nodes.items():
            if n.children == 0 and self.alloc.ref[n.block] == 1 and (best is None or n.last < best[1].last):
                best = (key, n)
        if best is None:
            return False
        key, n = best
        del self.nodes[key]
        del self.by_id[n.id]
        if n.parent:
            self.by_id[n.parent].children -= 1
        self.alloc.decref(n.block)
        return True

    def block_refs(self):
        refs = {}
        for n in self.nodes.values():
            refs[n.block] = refs.get(n.block, 0) + 1
        return refs


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
class Seq:
    __slots__ = ("id", "arrival", "prompt", "max_new", "out", "computed", "target", "table",
                 "first_token", "done", "preempts", "rejected", "cached_tokens")

    def __init__(self, id, arrival, prompt, max_new):
        prompt = list(prompt)
        if not math.isfinite(arrival) or arrival < 0:
            raise ValueError("arrival must be finite and nonnegative")
        if not prompt or not isinstance(max_new, (int, np.integer)) or max_new <= 0:
            raise ValueError("a request needs a nonempty prompt and positive integer max_new")
        self.id, self.arrival, self.prompt, self.max_new = id, arrival, prompt, max_new
        self.out, self.computed, self.target = [], 0, len(prompt)
        self.table = []
        self.first_token = self.done = None
        self.preempts = 0
        self.rejected = False
        self.cached_tokens = 0


class Engine:
    def __init__(self, model, n_blocks=256, B=8, max_seqs=16, budget=None, watermark=0.0,
                 prefix_cache=False, hook=None, debug=False):
        if any(not isinstance(x, (int, np.integer)) or x <= 0 for x in (n_blocks, B, max_seqs)):
            raise ValueError("n_blocks, B and max_seqs must be positive integers")
        if budget is not None and (not isinstance(budget, (int, np.integer)) or budget <= 0):
            raise ValueError("budget must be None or a positive integer")
        if not math.isfinite(watermark) or not 0 <= watermark < 1:
            raise ValueError("watermark must be finite and in [0, 1)")
        self.m, self.n_blocks, self.B = model, n_blocks, B
        self.max_seqs = max_seqs
        self.budget_cap = budget if budget else 10 ** 12
        self.wm = int(math.ceil(watermark * n_blocks))
        self.alloc = BlockAllocator(n_blocks)
        self.pool = KVPool(model, n_blocks, B)
        self.cache = PrefixCache(B, self.alloc) if prefix_cache else None
        self.hook, self.debug = hook, debug
        self.stats = {"iters": 0, "prefill_tokens": 0, "preemptions": 0, "hit_tokens": 0}

    def need(self, tokens):
        return -(-tokens // self.B)

    def _free_blocks(self, table):
        for b in table:
            self.alloc.decref(b)

    def _ensure_free(self, n):
        while self.alloc.num_free() < n:
            if self.cache is None or not self.cache.evict_one():
                return False
        return True

    def check_invariants(self, running):
        refs = {}
        for s in running:
            for b in s.table:
                refs[b] = refs.get(b, 0) + 1
        for b, c in (self.cache.block_refs().items() if self.cache else []):
            refs[b] = refs.get(b, 0) + c
        for b in range(self.n_blocks):
            assert self.alloc.ref[b] == refs.get(b, 0), f"block {b}: ref {self.alloc.ref[b]} vs holders {refs.get(b, 0)}"

    def run(self, reqs):
        """reqs: list of (id, arrival, prompt tokens, max_new). Returns the list of Seq."""
        seqs = [Seq(*r) for r in reqs]
        # Reject the whole input batch before allocating or mutating the prefix cache.
        for seq in seqs:
            if any(not isinstance(t, (int, np.integer)) or not 0 <= t < self.m.V for t in seq.prompt):
                raise ValueError("prompt token ids must be integers in the model vocabulary")
            if len(seq.prompt) + seq.max_new - 1 > len(self.m.P):
                raise ValueError("request exceeds the model's positional capacity")
        pending = sorted(seqs, key=lambda s: s.arrival)
        waiting, running = deque(), []
        ip, t, finished, rejected, n = 0, 0.0, 0, 0, len(seqs)
        B, st = self.B, self.stats

        def finish(s):
            nonlocal finished
            s.done = t
            running.remove(s)
            if self.cache:
                self.cache.register((s.prompt + s.out)[:s.computed], s.table)
            self._free_blocks(s.table)
            s.table = []
            finished += 1

        def preempt(v):
            running.remove(v)
            self._free_blocks(v.table)
            v.table, v.computed = [], 0
            v.target = len(v.prompt) + len(v.out)
            v.preempts += 1
            st["preemptions"] += 1
            waiting.appendleft(v)

        while finished + rejected < n:
            if not running and not waiting and ip < n and pending[ip].arrival > t:
                t = pending[ip].arrival
            while ip < n and pending[ip].arrival <= t:
                s = pending[ip]
                ip += 1
                if self.need(len(s.prompt) + s.max_new) > self.n_blocks - self.wm:
                    s.rejected = True
                    rejected += 1
                else:
                    waiting.append(s)
            if not running and not waiting:
                continue

            preempted_now = False
            decoders = []
            for s in list(running):
                if s not in running or s.computed < s.target:
                    continue
                if s.computed % B == 0:
                    while not self._ensure_free(1) and s in running:
                        preempt(running[-1])
                        preempted_now = True
                    if s not in running:
                        continue
                    s.table.append(self.alloc.alloc())
                decoders.append(s)
            budget = self.budget_cap - len(decoders)
            chunks = []
            for s in running:
                if s.computed < s.target and budget > 0:
                    c = min(s.target - s.computed, budget)
                    chunks.append((s, c))
                    budget -= c
            if not preempted_now:
                while waiting and len(running) < self.max_seqs and budget > 0:
                    s = waiting[0]
                    toks = s.prompt + s.out
                    matched = self.cache.match(toks, (len(toks) - 1) // B) if self.cache else []
                    for b in matched:
                        self.alloc.incref(b)                      # pin while we decide
                    new_needed = self.need(s.target) - len(matched)
                    if not self._ensure_free(new_needed + self.wm):
                        self._free_blocks(matched)
                        break                                     # strict FCFS: head blocks the queue
                    waiting.popleft()
                    s.table = matched + [self.alloc.alloc() for _ in range(new_needed)]
                    s.computed = len(matched) * B
                    s.cached_tokens += s.computed
                    st["hit_tokens"] += s.computed
                    running.append(s)
                    c = min(s.target - s.computed, budget)
                    chunks.append((s, c))
                    budget -= c
            pf = sum(c for _, c in chunks)
            assert pf + len(decoders) > 0, "engine made no progress"
            t += T0 + CP * pf + CD * len(decoders)
            st["iters"] += 1
            st["prefill_tokens"] += pf

            # ---- execute the model ----
            for s in decoders:
                logits = forward_chunk(self.m, self.pool, s.table, [s.out[-1]], s.computed)
                s.computed += 1
                s.out.append(int(np.argmax(logits)))
                if len(s.out) == s.max_new:
                    finish(s)
            for s, c in chunks:
                toks = s.prompt + s.out
                logits = forward_chunk(self.m, self.pool, s.table, toks[s.computed:s.computed + c], s.computed)
                s.computed += c
                if s.computed == s.target:
                    s.out.append(int(np.argmax(logits)))
                    if s.first_token is None:
                        s.first_token = t
                    if self.cache:
                        self.cache.register(toks[:s.target], s.table)
                    if len(s.out) == s.max_new:
                        finish(s)
            if self.debug:
                self.check_invariants(running)
            if self.hook:
                self.hook(self)
        self.end = t
        return seqs


# --------------------------------------------------------------------------
# Workloads
# --------------------------------------------------------------------------
def make_requests(seed, n, V=96, rate=30.0, sys_len=40, n_sys=3, share=0.6, ulo=8, uhi=60, nlo=4, nhi=20):
    r = np.random.default_rng(seed)
    sys_prompts = [list(map(int, r.integers(0, V, sys_len))) for _ in range(n_sys)]
    t = 0.0
    out = []
    for i in range(n):
        t += r.exponential(1 / rate)
        user = list(map(int, r.integers(0, V, int(r.integers(ulo, uhi)))))
        prompt = (sys_prompts[int(r.integers(n_sys))] + user) if r.random() < share else user
        out.append((i, t, prompt, int(r.integers(nlo, nhi))))
    return out


_REF = {}


def reference(model, reqs):
    res = {}
    for (i, _, prompt, k) in reqs:
        key = (id(model), tuple(prompt), k)
        if key not in _REF:
            _REF[key] = model.ref_generate(prompt, k)
        res[i] = _REF[key]
    return res


def mismatches(model, reqs, seqs):
    ref = reference(model, reqs)
    return sum(1 for s in seqs if not s.rejected and s.out != ref[s.id])


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------
def selftests():
    t0 = time.time()
    n_ok = 0

    def ok(name):
        nonlocal n_ok
        n_ok += 1
        print(f"  PASS {n_ok:2d}  {name}")

    M = TinyLM()
    r = np.random.default_rng(0)

    # 1 causality of the reference model
    a = list(map(int, r.integers(0, 96, 30)))
    b = a[:20] + list(map(int, r.integers(0, 96, 10)))
    assert np.array_equal(M.forward_full(a)[:20], M.forward_full(b)[:20])
    ok("reference model is causal: changing later tokens leaves earlier logits bit-identical")

    # 2 KV-cached incremental forward == full forward
    worst = 0.0
    for B in (1, 4, 8, 16):
        toks = list(map(int, r.integers(0, 96, 37)))
        pool = KVPool(M, 64, B)
        table = list(range(-(-len(toks) // B)))
        full = M.forward_full(toks)
        pos = 0
        for c in (5, 1, 1, 12, 1, 17):                       # mixed chunk sizes incl. single-token decode
            lg = forward_chunk(M, pool, table, toks[pos:pos + c], pos)
            worst = max(worst, float(np.abs(lg - full[pos + c - 1]).max()))
            pos += c
        assert pos == len(toks)
    assert worst < 1e-10, worst
    ok(f"paged chunked forward equals the full forward for B = 1, 4, 8, 16 (max |diff| {worst:.1e})")

    # 3 decode attention (block walk) equals gather attention
    pool = KVPool(M, 32, 4)
    table = [5, 2, 9, 0, 7]
    toks = list(map(int, r.integers(0, 96, 18)))
    forward_chunk(M, pool, table, toks, 0)
    q = r.standard_normal((M.H, M.D))
    w1 = attn_paged_decode(q, pool, 0, table, 18)
    w2 = attn_gather_chunk(q[None], pool, 0, table, 17)[0]
    assert np.abs(w1 - w2).max() < 1e-12
    ok("decode attention that walks scattered blocks equals gathered attention")

    reqs = make_requests(1, 24)
    ref = reference(M, reqs)

    # 4 single request, several block sizes and chunk budgets
    for B in (2, 5, 8, 16):
        for bud in (None, 3, 16):
            eng = Engine(M, n_blocks=400, B=B, budget=bud, debug=True)
            s = eng.run([reqs[3]])[0]
            assert s.out == ref[3], (B, bud)
    ok("one request matches the reference for B in (2, 5, 8, 16) and chunk budgets (none, 3, 16)")

    # 5 many concurrent requests, roomy pool
    eng = Engine(M, n_blocks=2000, B=8, debug=True)
    seqs = eng.run(reqs)
    assert mismatches(M, reqs, seqs) == 0 and eng.stats["preemptions"] == 0
    ok("24 concurrent requests, continuous batching: every output equals the reference")

    # 6 tiny pool forces preemption by recomputation; outputs unchanged
    eng = Engine(M, n_blocks=26, B=8, debug=True)
    seqs = eng.run(reqs)
    assert eng.stats["preemptions"] > 0 and mismatches(M, reqs, seqs) == 0
    ok(f"tiny pool: {eng.stats['preemptions']} preemptions with recomputation, outputs still equal the reference")

    # 7 chunked prefill with a small budget
    eng = Engine(M, n_blocks=2000, B=8, budget=12, debug=True)
    assert mismatches(M, reqs, eng.run(reqs)) == 0
    ok("chunked prefill (budget 12 tokens per iteration): outputs equal the reference")

    # 8 prefix cache on
    eng = Engine(M, n_blocks=2000, B=8, prefix_cache=True, debug=True)
    seqs = eng.run(reqs)
    assert mismatches(M, reqs, seqs) == 0 and eng.stats["hit_tokens"] > 0
    base = Engine(M, n_blocks=2000, B=8)
    base.run(reqs)
    assert eng.stats["prefill_tokens"] < base.stats["prefill_tokens"]
    assert eng.stats["prefill_tokens"] + eng.stats["hit_tokens"] == base.stats["prefill_tokens"]
    ok(f"prefix cache: outputs equal the reference; {eng.stats['hit_tokens']} prompt tokens served from cached blocks")

    # 9 prefix cache + tiny pool + chunking, reference reference-equal, invariants each iteration
    for B, nb, bud in ((8, 30, None), (4, 60, 10), (16, 14, 20)):
        eng = Engine(M, n_blocks=nb, B=B, budget=bud, prefix_cache=True, debug=True)
        seqs = eng.run(reqs)
        assert mismatches(M, reqs, seqs) == 0
    ok("cache + pressure + chunking in three configurations: outputs equal the reference, refcounts exact every iteration")

    # 10 no leaks: after clearing the cache every block is free
    eng = Engine(M, n_blocks=40, B=8, prefix_cache=True)
    eng.run(reqs)
    while eng.cache.evict_one():
        pass
    assert eng.alloc.num_free() == 40 and not eng.cache.nodes
    ok("after the run and a full cache flush every block is back on the free list")

    # 11 engine timing equals the Chapter 44 simulator (cache off) request by request
    for B, nb, bud in ((8, 2000, None), (8, 26, None), (8, 2000, 12), (16, 30, 24)):
        eng = Engine(M, n_blocks=nb, B=B, budget=bud, watermark=0.0)
        seqs = {s.id: s for s in eng.run(reqs)}
        sim = [c44.Req(i, a, len(p), k) for (i, a, p, k) in reqs]
        st = c44.simulate_continuous(sim, n_blocks=nb, B=B, max_seqs=16, max_batched_tokens=bud, watermark=0.0)
        for q in sim:
            s = seqs[q.id]
            assert q.rejected == s.rejected
            if not q.rejected:
                assert abs(q.first_token - s.first_token) < 1e-9 and abs(q.done - s.done) < 1e-9, (B, nb, bud, q.id)
        assert st["preemptions"] == eng.stats["preemptions"] and st["iters"] == eng.stats["iters"]
    ok("with the cache off, first-token and finish times, iterations and preemptions equal the Chapter 44 simulator exactly (4 configurations)")

    # 12 block-trie match equals the Chapter 45 radix match rounded down to a block
    for seed in range(10):
        rr = np.random.default_rng(seed)
        B = int(rr.choice([2, 4, 8]))
        alloc = BlockAllocator(500)
        pc = PrefixCache(B, alloc)
        rc = c45.RadixCache(10 ** 9)
        for _ in range(40):
            q = tuple(int(x) for x in rr.integers(0, 3, int(rr.integers(1, 40))))
            m_trie = len(pc.match(q, len(q) // B)) * B
            m_rad = rc.match_length(q)
            assert m_trie == (m_rad // B) * B, (m_trie, m_rad, B)
            table = [alloc.alloc() for _ in range(-(-len(q) // B))]
            pc.register(q, table)
            rc.insert(q)
            for b in table:
                alloc.decref(b)
    ok("the engine's block trie matches the Chapter 45 radix tree rounded down to whole blocks (10 sessions x 40 sequences)")

    # 13 the equality check has teeth: corrupting one cached block is detected
    corrupted = []

    def hook(e):
        if e.cache and e.cache.nodes and not corrupted:
            for n in e.cache.nodes.values():
                e.pool.K[:, n.block] *= -3.0
                e.pool.V[:, n.block] += 5.0
                corrupted.append(n.block)

    shared = list(map(int, np.random.default_rng(5).integers(0, 96, 40)))
    rq = [(0, 0.0, shared + [1, 2, 3, 4, 5], 10), (1, 5.0, shared + [9, 8, 7, 6, 5], 10)]
    eng = Engine(M, n_blocks=200, B=8, prefix_cache=True, hook=hook)
    seqs = eng.run(rq)
    assert corrupted and mismatches(M, rq, seqs) >= 1
    ok("sanity check of the test itself: corrupting the cached KV blocks makes outputs disagree with the reference")

    # 14 an oversized request is rejected; the others complete correctly
    eng = Engine(M, n_blocks=20, B=8)
    seqs = eng.run([(0, 0.0, list(map(int, r.integers(0, 96, 150))), 50), (1, 0.0, reqs[1][2], reqs[1][3])])
    assert seqs[0].rejected and seqs[1].out == ref[1]
    ok("a request larger than the pool is rejected; the other one still matches the reference")

    # 15 outputs do not depend on arrival times (batch composition)
    shifted = [(i, a * 0.1, p, k) for (i, a, p, k) in reqs]
    s1 = {s.id: s.out for s in Engine(M, n_blocks=60, B=8, prefix_cache=True).run(reqs)}
    s2 = {s.id: s.out for s in Engine(M, n_blocks=60, B=8, prefix_cache=True).run(shifted)}
    assert s1 == s2
    ok("compressing arrival times (a different batch composition) changes no output")

    print(f"ALL PASS   ({time.time() - t0:.0f} s)  [{n_ok} tests]")


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def experiments():
    M = TinyLM()
    print("model: %d layers, d_model %d, %d heads, vocab %d (random weights, float64)" % (M.L, M.d, M.H, M.V))
    reqs = make_requests(1, 48)
    ref = reference(M, reqs)
    print("workload: 48 requests, 60%% start with one of 3 shared 40-token prompts; prompt tokens %d, output tokens %d" %
          (sum(len(p) for _, _, p, _ in reqs), sum(k for *_, k in reqs)))

    print("\n=== Experiment 1: correctness matrix (every row is checked against the no-cache reference) ===")
    print(f"{'configuration':>44} {'mismatches':>10} {'iters':>6} {'preempt':>8} {'prefill tok':>12} {'cache hit tok':>14} {'sim end (s)':>11}")
    cfgs = [
        ("B=8, roomy pool, no cache", dict(n_blocks=2000, B=8)),
        ("B=8, tiny pool (26 blocks), no cache", dict(n_blocks=26, B=8)),
        ("B=8, chunked prefill (budget 12)", dict(n_blocks=2000, B=8, budget=12)),
        ("B=8, roomy, prefix cache", dict(n_blocks=2000, B=8, prefix_cache=True)),
        ("B=8, tiny pool, prefix cache", dict(n_blocks=26, B=8, prefix_cache=True)),
        ("B=4, 60 blocks, budget 10, prefix cache", dict(n_blocks=60, B=4, budget=10, prefix_cache=True)),
        ("B=16, 14 blocks, budget 20, prefix cache", dict(n_blocks=14, B=16, budget=20, prefix_cache=True)),
    ]
    for name, kw in cfgs:
        eng = Engine(M, **kw)
        seqs = eng.run(reqs)
        print(f"{name:>44} {mismatches(M, reqs, seqs):10d} {eng.stats['iters']:6d} {eng.stats['preemptions']:8d} "
              f"{eng.stats['prefill_tokens']:12d} {eng.stats['hit_tokens']:14d} {eng.end:11.2f}")

    print("\n=== Experiment 2: what the prefix cache buys (roomy pool and tiny pool) ===")
    print(f"{'configuration':>30} {'prefill tok':>12} {'preempt':>8} {'TTFT p50 (ms)':>14} {'TTFT p99 (ms)':>14} {'e2e p50 (s)':>12}")
    for name, kw in (("roomy, no cache", dict(n_blocks=2000, B=8)), ("roomy, cache", dict(n_blocks=2000, B=8, prefix_cache=True)),
                     ("tiny pool, no cache", dict(n_blocks=26, B=8)), ("tiny pool, cache", dict(n_blocks=26, B=8, prefix_cache=True))):
        eng = Engine(M, **kw)
        seqs = eng.run(reqs)
        tt = np.array([s.first_token - s.arrival for s in seqs]) * 1000
        e2 = np.array([s.done - s.arrival for s in seqs])
        print(f"{name:>30} {eng.stats['prefill_tokens']:12d} {eng.stats['preemptions']:8d} {np.percentile(tt, 50):14.1f} {np.percentile(tt, 99):14.1f} {np.percentile(e2, 50):12.3f}")

    print("\n=== Experiment 3: block size and cache hits (roomy pool, 40-token shared prompts) ===")
    print(f"{'B':>4} {'hit tokens':>11} {'prefill tokens':>15} {'floor(40/B)*B':>14}")
    for B in (4, 8, 16, 32):
        eng = Engine(M, n_blocks=4000, B=B, prefix_cache=True)
        eng.run(reqs)
        print(f"{B:4d} {eng.stats['hit_tokens']:11d} {eng.stats['prefill_tokens']:15d} {(40 // B) * B:14d}")

    print("\n=== Experiment 4: mutation check (a corrupted cached block must be caught) ===")
    done = []

    def hook(e):
        if e.cache and e.cache.nodes and not done:
            for n in e.cache.nodes.values():
                e.pool.K[:, n.block] *= -3.0
                e.pool.V[:, n.block] += 5.0
                done.append(n.block)
    eng = Engine(M, n_blocks=2000, B=8, prefix_cache=True, hook=hook)
    seqs = eng.run(reqs)
    print(f"corrupted {len(done)} cached blocks (K x -3, V + 5) the first time the cache is non-empty; requests whose output differs from the reference: {mismatches(M, reqs, seqs)} of {len(seqs)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tests":
        selftests()
    else:
        selftests()
        experiments()
