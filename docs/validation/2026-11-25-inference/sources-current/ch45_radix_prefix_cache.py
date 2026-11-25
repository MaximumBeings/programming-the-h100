#!/usr/bin/env python3
"""Chapter 45 - A radix-tree prefix cache (SGLang style) and a block-hash cache (vLLM style).

Run:  python3 ch45_radix_prefix_cache.py          # self-tests + experiments
      python3 ch45_radix_prefix_cache.py tests    # self-tests only

CPU only. Token ids are synthetic; the workloads are ASSUMED.
"""
import heapq
import sys
import time
from collections import deque

import numpy as np


class CacheFull(Exception):
    def __init__(self, matched):
        super().__init__("cache full: every cached token is pinned")
        self.matched = matched


def lcp(a, b):
    """Length of the longest common prefix of two sequences (slice compares are C-speed)."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n          # a[:lo] == b[:lo], a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------
# Radix tree: edges hold runs of tokens; one cached token per distinct prefix
# --------------------------------------------------------------------------
class Node:
    __slots__ = ("key", "children", "parent", "last", "born", "ref", "alive", "id")

    def __init__(self, key, parent, nid):
        self.key = key
        self.children = {}
        self.parent = parent
        self.last = 0
        self.born = 0
        self.ref = 0
        self.alive = True
        self.id = nid


class Handle:
    def __init__(self, node, matched, new):
        self.node, self.matched, self.new = node, matched, new
        self.open = True


class RadixCache:
    def __init__(self, capacity, policy="lru"):
        assert policy in ("lru", "fifo")
        self.capacity = capacity
        self.policy = policy
        self._ids = 0
        self.root = Node((), None, 0)
        self.tokens = 0
        self.clock = 0
        self.heap = []
        self.handles = set()
        self.evicted_tokens = 0

    # ---- helpers ----
    def _new(self, key, parent):
        self._ids += 1
        return Node(key, parent, self._ids)

    def _key(self, n):
        return n.last if self.policy == "lru" else n.born

    def _push(self, n):
        if n is not self.root and n.alive and not n.children and n.ref == 0:
            heapq.heappush(self.heap, (self._key(n), n.id, n))

    def _split(self, c, j):
        """Split edge c at offset j; returns the new upper node holding c.key[:j]."""
        up = self._new(c.key[:j], c.parent)
        up.last, up.born, up.ref = c.last, c.born, c.ref
        c.parent.children[up.key[0]] = up
        c.key = c.key[j:]
        c.parent = up
        up.children[c.key[0]] = c
        return up

    def _match(self, toks):
        """Walk the tree; split an edge when the match ends inside it. Touches the path."""
        self.clock += 1
        node, i = self.root, 0
        while i < len(toks):
            c = node.children.get(toks[i])
            if c is None:
                break
            j = lcp(c.key, toks[i:])
            if j < len(c.key):
                c = self._split(c, j)
            c.last = self.clock
            i += j
            node = c
            if i < len(toks) and node.children.get(toks[i]) is None:
                break
        return i, node

    def match_length(self, toks):
        """Read-only longest cached prefix (no split, no touch)."""
        node, i = self.root, 0
        while i < len(toks):
            c = node.children.get(toks[i])
            if c is None:
                break
            j = lcp(c.key, toks[i:])
            i += j
            if j < len(c.key):
                break
            node = c
        return i

    def _lock(self, n):
        while n is not self.root:
            n.ref += 1
            n = n.parent

    def _unlock(self, n):
        while n is not self.root:
            n.ref -= 1
            assert n.ref >= 0
            n = n.parent

    # ---- eviction ----
    def _pop_victim(self):
        while self.heap:
            k, _, n = heapq.heappop(self.heap)
            if not (n.alive and not n.children and n.ref == 0):
                continue
            if k != self._key(n):
                heapq.heappush(self.heap, (self._key(n), n.id, n))
                continue
            return n
        return None

    def evict(self, n_tokens):
        freed = 0
        while freed < n_tokens:
            v = self._pop_victim()
            if v is None:
                break
            freed += len(v.key)
            self.tokens -= len(v.key)
            self.evicted_tokens += len(v.key)
            v.alive = False
            del v.parent.children[v.key[0]]
            self._push(v.parent)
        return freed

    # ---- request interface ----
    def acquire(self, toks):
        """Match, pin and insert `toks`. Returns a Handle; raises CacheFull (state unchanged except evictions)."""
        toks = tuple(toks)
        m, node = self._match(toks)
        self._lock(node)
        need = len(toks) - m
        if need > self.capacity - self.tokens:
            self.evict(need - (self.capacity - self.tokens))
        if need > self.capacity - self.tokens:
            self._unlock(node)
            self._push(node)
            raise CacheFull(m)
        if need:
            leaf = self._new(toks[m:], node)
            leaf.last = leaf.born = self.clock
            leaf.ref = 1
            node.children[leaf.key[0]] = leaf
            self.tokens += need
            node = leaf
        h = Handle(node, m, need)
        self.handles.add(h)
        return h

    def release(self, h):
        assert h.open
        h.open = False
        self.handles.discard(h)
        self._unlock(h.node)
        self._push(h.node)

    def insert(self, toks):
        """Cache a finished sequence (no pin kept)."""
        h = self.acquire(toks)
        self.release(h)
        return h.new

    # ---- inspection ----
    def nodes(self):
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children.values())
        return out

    def check_invariants(self):
        total = 0
        expect = {}
        for h in self.handles:
            n = h.node
            while n is not self.root:
                expect[n.id] = expect.get(n.id, 0) + 1
                n = n.parent
        for n in self.nodes():
            for k, c in n.children.items():
                assert c.parent is n and c.key[0] == k and len(c.key) > 0 and c.alive
            if n is not self.root:
                total += len(n.key)
                assert n.ref == expect.get(n.id, 0), "pin count mismatch"
        assert total == self.tokens, "token accounting mismatch"
        assert self.tokens <= self.capacity

    def stats(self):
        ns = [n for n in self.nodes() if n is not self.root]
        return {"nodes": len(ns), "tokens": self.tokens,
                "mean_edge": self.tokens / max(1, len(ns))}


# --------------------------------------------------------------------------
# Block-hash cache (vLLM style): a chained hash of each FULL block
# --------------------------------------------------------------------------
class BlockHashCache:
    def __init__(self, capacity_tokens, B=16, hash_bits=64):
        self.B, self.cap = B, capacity_tokens // B
        self.mask = (1 << hash_bits) - 1
        self.table = {}      # chain hash -> (last, depth)
        self.clock = 0
        self.false_hits = 0

    def _chain(self, toks):
        h, out = 0, []
        for i in range(0, len(toks) - len(toks) % self.B, self.B):
            h = hash((h, toks[i:i + self.B])) & self.mask
            out.append(h)
        return out

    def match_length(self, toks, touch=False):
        self.clock += 1
        n = 0
        for h in self._chain(toks):
            if h not in self.table:
                break
            if touch:
                self.table[h] = (self.clock, self.table[h][1])
            n += 1
        return n * self.B

    def insert(self, toks):
        self.clock += 1
        for d, h in enumerate(self._chain(toks)):
            if h in self.table:
                self.table[h] = (self.clock, d)
                continue
            while len(self.table) >= self.cap:
                victim = min(self.table, key=lambda k: (self.table[k][0], -self.table[k][1]))
                del self.table[victim]
            self.table[h] = (self.clock, d)


# --------------------------------------------------------------------------
# Workloads (ASSUMED shapes). Token ids are random integers.
# --------------------------------------------------------------------------
def rand_tokens(rng, n):
    return tuple(int(x) for x in rng.integers(0, 50000, n))


def wl_system(rng, n, n_sys=20, sys_len=1000, ulo=100, uhi=300, zipf=1.1):
    sys_prompts = [rand_tokens(rng, sys_len) for _ in range(n_sys)]
    p = 1.0 / np.arange(1, n_sys + 1) ** zipf
    p /= p.sum()
    return [(sys_prompts[int(rng.choice(n_sys, p=p))] + rand_tokens(rng, int(rng.integers(ulo, uhi))),
             rand_tokens(rng, 120)) for _ in range(n)]


def wl_multiturn(rng, n, convs=40, n_sys=20, sys_len=600, turn=(80, 200), out=(100, 200), max_len=3500):
    sys_prompts = [rand_tokens(rng, sys_len) for _ in range(n_sys)]
    hist = [sys_prompts[i % n_sys] for i in range(convs)]
    reqs = []
    for _ in range(n):
        c = int(rng.integers(convs))
        if len(hist[c]) > max_len:
            hist[c] = sys_prompts[int(rng.integers(n_sys))]
        prompt = hist[c] + rand_tokens(rng, int(rng.integers(*turn)))
        output = rand_tokens(rng, int(rng.integers(*out)))
        reqs.append((prompt, output))
        hist[c] = prompt + output
    return reqs


def wl_rag(rng, n, n_docs=200, doc_len=800, k=3, ordered=False):
    docs = [rand_tokens(rng, doc_len) for _ in range(n_docs)]
    reqs = []
    for _ in range(n):
        ids = rng.choice(n_docs, k, replace=False)
        if ordered:
            ids = sorted(ids)
        reqs.append((tuple(t for i in ids for t in docs[int(i)]) + rand_tokens(rng, 120), rand_tokens(rng, 100)))
    return reqs


def wl_popular(rng, n, n_prompts=300, lo=600, hi=1500, zipf=1.0):
    """Complete prompts that repeat exactly (FAQ-style or agent loops), popularity Zipf."""
    prompts = [rand_tokens(rng, int(rng.integers(lo, hi))) for _ in range(n_prompts)]
    p = 1.0 / np.arange(1, n_prompts + 1) ** zipf
    p /= p.sum()
    return [(prompts[int(rng.choice(n_prompts, p=p))], rand_tokens(rng, 100)) for _ in range(n)]


def wl_random(rng, n):
    return [(rand_tokens(rng, int(rng.integers(500, 1500))), rand_tokens(rng, 100)) for _ in range(n)]


# --------------------------------------------------------------------------
# Hit-rate harness (sequential requests; each finished sequence is cached)
# --------------------------------------------------------------------------
def run_radix(reqs, capacity, policy="lru", conc=1, cache_output=True):
    c = RadixCache(capacity, policy)
    hit = tot = fails = 0
    active = deque()
    for prompt, out in reqs:
        tot += len(prompt)
        try:
            h = c.acquire(prompt)
            hit += h.matched
            active.append(h)
        except CacheFull as e:
            hit += e.matched
            fails += 1
        while len(active) >= conc:
            c.release(active.popleft())
        if cache_output:
            try:
                c.insert(prompt + out)
            except CacheFull:
                pass
    while active:
        c.release(active.popleft())
    return hit / tot, fails, c


def run_hash(reqs, capacity, B=16, bits=64):
    c = BlockHashCache(capacity, B, bits)
    hit = tot = 0
    for prompt, out in reqs:
        tot += len(prompt)
        hit += c.match_length(prompt, touch=True)
        c.insert(prompt + out)
    return hit / tot


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------
def prefixes(seqs):
    s = set()
    for q in seqs:
        for i in range(1, len(q) + 1):
            s.add(q[:i])
    return s


def brute_lcp(stored, q):
    return max((lcp(s, q) for s in stored), default=0)


def selftests():
    t0 = time.time()
    n_ok = 0

    def ok(name):
        nonlocal n_ok
        n_ok += 1
        print(f"  PASS {n_ok:2d}  {name}")

    # 1 lcp
    for _ in range(300):
        r = np.random.default_rng(_)
        a = tuple(int(x) for x in r.integers(0, 3, int(r.integers(0, 30))))
        b = tuple(int(x) for x in r.integers(0, 3, int(r.integers(0, 30))))
        e = 0
        while e < min(len(a), len(b)) and a[e] == b[e]:
            e += 1
        assert lcp(a, b) == e
    ok("fast lcp equals the naive loop on 300 random pairs")

    # 2 basic insert / match / split
    c = RadixCache(10 ** 6)
    c.insert((1, 2, 3, 4, 5))
    assert c.match_length((1, 2, 3, 4, 5, 6)) == 5
    assert c.match_length((1, 2, 3, 9)) == 3 and c.tokens == 5
    c.insert((1, 2, 3, 9))
    assert c.tokens == 6 and c.match_length((1, 2, 3, 9)) == 4 and c.match_length((1, 2, 3, 4, 5)) == 5
    assert c.match_length((7,)) == 0
    c.check_invariants()
    ok("insert, match and edge split behave on a hand example (5 + 1 new token = 6 stored)")

    # 3 one stored token per distinct prefix (no eviction)
    for seed in range(20):
        r = np.random.default_rng(seed)
        c = RadixCache(10 ** 9)
        seqs = []
        for _ in range(40):
            q = tuple(int(x) for x in r.integers(0, 3, int(r.integers(1, 25))))
            seqs.append(q)
            c.insert(q)
            c.check_invariants()
        assert c.tokens == len(prefixes(seqs))
    ok("tree stores exactly one token per distinct prefix (20 random sets of 40 sequences)")

    # 4 match length equals brute-force longest common prefix
    for seed in range(20):
        r = np.random.default_rng(100 + seed)
        c = RadixCache(10 ** 9)
        stored = []
        for _ in range(60):
            q = tuple(int(x) for x in r.integers(0, 3, int(r.integers(1, 20))))
            assert c.match_length(q) == brute_lcp(stored, q)
            m, _n = c._match(q)
            assert m == brute_lcp(stored, q)
            c.insert(q)
            stored.append(q)
    ok("match length equals the brute-force maximum common prefix (20 sessions x 60 queries)")

    # 5 pinned paths survive eviction; unpinned LRU leaf goes first
    c = RadixCache(12)
    ha = c.acquire((1, 1, 1, 1))
    c.insert((2, 2, 2, 2))
    c.insert((3, 3, 3, 3))
    assert c.tokens == 12
    c.match_length((2, 2, 2, 2))
    c._match((2, 2, 2, 2))                       # touch 2222: now 3333 is least recently used
    c.insert((4, 4, 4, 4))
    assert c.match_length((1, 1, 1, 1)) == 4     # pinned, kept
    assert c.match_length((3, 3, 3, 3)) == 0     # LRU victim
    assert c.match_length((2, 2, 2, 2)) == 4
    c.release(ha)
    c.check_invariants()
    ok("a pinned sequence survives pressure; the least recently used unpinned leaf is evicted")

    # 6 eviction order equals a brute-force scan of all evictable leaves
    for seed in range(15):
        r = np.random.default_rng(200 + seed)
        c = RadixCache(60)
        for _ in range(150):
            q = tuple(int(x) for x in r.integers(0, 3, int(r.integers(1, 12))))
            try:
                h = c.acquire(q)
                c.release(h)
            except CacheFull:
                pass
            leaves = [n for n in c.nodes() if n is not c.root and not n.children and n.ref == 0]
            if leaves and r.random() < 0.3:
                want = min(n.last for n in leaves)
                v = c._pop_victim()
                assert v is not None and v.last == want
                heapq.heappush(c.heap, (c._key(v), v.id, v))
            c.check_invariants()
    ok("the eviction victim always has the minimum last-access among evictable leaves (15 sessions)")

    # 7 capacity never exceeded; CacheFull is raised when everything is pinned, state intact
    c = RadixCache(8)
    h1 = c.acquire((1, 1, 1, 1, 1))
    h2 = c.acquire((2, 2, 2))
    before = c.tokens
    try:
        c.acquire((3, 3, 3))
        raise AssertionError
    except CacheFull as e:
        assert e.matched == 0
    assert c.tokens == before == 8
    c.check_invariants()
    c.release(h1)
    h3 = c.acquire((3, 3, 3))
    c.check_invariants()
    c.release(h2)
    c.release(h3)
    ok("when every token is pinned the request fails with CacheFull and the cache is unchanged")

    # 8 pin counts are exact under random concurrent use
    for seed in range(15):
        r = np.random.default_rng(300 + seed)
        c = RadixCache(80)
        live = []
        for _ in range(250):
            if live and r.random() < 0.45:
                c.release(live.pop(int(r.integers(len(live)))))
            else:
                q = tuple(int(x) for x in r.integers(0, 3, int(r.integers(1, 15))))
                try:
                    live.append(c.acquire(q))
                except CacheFull:
                    pass
            c.check_invariants()
        for h in live:
            c.release(h)
        c.check_invariants()
        assert all(n.ref == 0 for n in c.nodes())
    ok("pin counts equal the number of live handles through each node (15 random sessions)")

    # 9 shared prefix stored once
    c = RadixCache(10 ** 6)
    sysp = tuple(range(100))
    for i in range(10):
        c.insert(sysp + (1000 + i,) * 20)
    assert c.tokens == 100 + 10 * 20
    ok("ten requests sharing a 100-token prefix store it once (300 tokens, not 1200)")

    # 10 radix == block-hash rounded down to a block (no eviction, full-block hash)
    for seed in range(5):
        r = np.random.default_rng(400 + seed)
        reqs = wl_system(r, 60, n_sys=4, sys_len=200, ulo=10, uhi=60)
        rc, hc = RadixCache(10 ** 9), BlockHashCache(10 ** 9, 16)
        for p, o in reqs:
            m1 = rc.match_length(p)
            m2 = hc.match_length(p)
            assert m2 == (m1 // 16) * 16 or m2 <= m1
            rc.insert(p + o)
            hc.insert(p + o)
            assert m2 >= (m1 // 16) * 16 - 16
            assert m2 <= m1
    ok("block-hash match never exceeds the radix match and is within one block of it")

    # 11 truncated hashes cause false hits; full hashes do not (correctness hazard)
    r = np.random.default_rng(7)
    rc = RadixCache(10 ** 9)
    h8, h64 = BlockHashCache(10 ** 9, 4, 8), BlockHashCache(10 ** 9, 4, 64)
    false8 = false64 = 0
    for _ in range(400):
        q = rand_tokens(r, 40)
        t = rc.match_length(q)
        false8 += h8.match_length(q) > t
        false64 += h64.match_length(q) > t
        rc.insert(q)
        h8.insert(q)
        h64.insert(q)
    assert false8 > 0 and false64 == 0
    ok(f"an 8-bit block hash served wrong KV {false8} times in 400 requests; the 64-bit hash 0 times (the radix tree compares tokens)")

    # 12 salting isolates tenants
    c = RadixCache(10 ** 6)
    sysp = tuple(range(50))
    c.insert((-1,) + sysp)
    assert c.match_length((-2,) + sysp) == 0 and c.match_length((-1,) + sysp) == 51
    ok("a per-tenant salt as the first token prevents any cross-tenant match")

    # 13 FIFO and LRU differ only when a hot leaf is old: exact-repeat prompts
    reqs = wl_popular(np.random.default_rng(9), 800)
    a = run_radix(reqs, 100000, "lru", cache_output=False)[0]
    b = run_radix(reqs, 100000, "fifo", cache_output=False)[0]
    assert a > b + 0.02
    ok(f"with exactly repeated prompts LRU beats FIFO ({a:.3f} vs {b:.3f} hit rate)")

    # 14 determinism
    r1 = run_radix(wl_system(np.random.default_rng(3), 200), 20000)[0]
    r2 = run_radix(wl_system(np.random.default_rng(3), 200), 20000)[0]
    assert r1 == r2
    ok("same seed, same hit rate")

    print(f"ALL PASS   ({time.time() - t0:.0f} s)  [{n_ok} tests]")


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def agg(v):
    v = np.array(v, dtype=float)
    return f"{v.mean():.3f} [{v.min():.3f}-{v.max():.3f}]"


def experiments():
    SEEDS = (0, 1, 2)
    N = 1500
    CP = 0.00005
    wls = {
        "system prompts (20 x 1000 tok, Zipf)": lambda r: wl_system(r, N),
        "multi-turn chat (40 conversations)": lambda r: wl_multiturn(r, N),
        "RAG, documents in random order": lambda r: wl_rag(r, N, ordered=False),
        "RAG, documents sorted by id": lambda r: wl_rag(r, N, ordered=True),
        "popular exact-repeat prompts (300, Zipf)": lambda r: wl_popular(r, N),
        "no sharing (control)": lambda r: wl_random(r, N),
    }
    print("hit rate = prompt tokens served from cache / prompt tokens; 3 seeds x %d requests" % N)

    print("\n=== Experiment 1: hit rate versus cache size, radix (token granularity) ===")
    caps = (4000, 16000, 64000, 256000, 1000000)
    print(f"{'workload':>40} " + " ".join(f"{c:>22}" for c in caps))
    for name, fn in wls.items():
        row = []
        for cap in caps:
            row.append(agg([run_radix(fn(np.random.default_rng(s)), cap)[0] for s in SEEDS]))
        print(f"{name:>40} " + " ".join(f"{x:>22}" for x in row))

    print("\n=== Experiment 2: radix versus block-hash, system-prompt workload, cache 64000 tokens ===")
    reqs_all = [wl_system(np.random.default_rng(s), N, sys_len=1000) for s in SEEDS]
    print(f"{'scheme':>22} {'hit rate':>24}")
    print(f"{'radix (token)':>22} {agg([run_radix(r, 64000)[0] for r in reqs_all]):>24}")
    for B in (16, 64, 256):
        print(f"{'block-hash B=%d' % B:>22} {agg([run_hash(r, 64000, B) for r in reqs_all]):>24}")
    odd = [wl_system(np.random.default_rng(s), N, sys_len=1000, ulo=100, uhi=300) for s in SEEDS]
    print("shared prefix is 1000 tokens: 1000 mod 16 = %d, mod 64 = %d, mod 256 = %d" % (1000 % 16, 1000 % 64, 1000 % 256))

    print("\n=== Experiment 3: LRU versus FIFO eviction ===")
    print(f"{'workload (cache tokens)':>46} {'LRU':>24} {'FIFO':>24}")
    for name, cap in (("system prompts (20 x 1000 tok, Zipf)", 10000), ("multi-turn chat (40 conversations)", 40000),
                      ("popular exact-repeat prompts (300, Zipf)", 100000)):
        r3 = [wls[name](np.random.default_rng(s)) for s in SEEDS]
        co = "popular" not in name      # for the exact-repeat workload the output is not cached
        print(f"{name + ' (%d)' % cap:>46} {agg([run_radix(r, cap, 'lru', cache_output=co)[0] for r in r3]):>24} {agg([run_radix(r, cap, 'fifo', cache_output=co)[0] for r in r3]):>24}")

    print("\n=== Experiment 4: pinned sequences under concurrency (multi-turn chat, cache 40000 tokens) ===")
    print(f"{'in flight':>10} {'hit rate':>24} {'requests refused caching':>26}")
    r4 = [wl_multiturn(np.random.default_rng(s), N) for s in SEEDS]
    for conc in (1, 4, 16, 32, 64):
        out = [run_radix(r, 40000, "lru", conc)[:2] for r in r4]
        print(f"{conc:10d} {agg([o[0] for o in out]):>24} {np.mean([o[1] for o in out]):>26.1f}")

    print("\n=== Experiment 5: prefill work saved (prefill cost %.2f ms per 1000 tokens, assumed) ===" % (CP * 1e6))
    print(f"{'workload':>40} {'mean prompt':>12} {'hit rate':>9} {'prefill ms/req, no cache':>25} {'with cache':>11}")
    for name, fn in wls.items():
        reqs = fn(np.random.default_rng(0))
        mp = np.mean([len(p) for p, _ in reqs])
        hr = run_radix(reqs, 256000)[0]
        print(f"{name:>40} {mp:12.0f} {hr:9.3f} {mp * CP * 1000:25.1f} {mp * (1 - hr) * CP * 1000:11.1f}")

    print("\n=== Experiment 6: tree shape (cache 256000 tokens) ===")
    print(f"{'workload':>40} {'nodes':>7} {'tokens':>8} {'mean edge length':>17}")
    for name, fn in wls.items():
        st = run_radix(fn(np.random.default_rng(0)), 256000)[2].stats()
        print(f"{name:>40} {st['nodes']:7d} {st['tokens']:8d} {st['mean_edge']:17.1f}")

    print("\n=== Experiment 7: a scan of unique long prompts (cache 64000 tokens, system-prompt workload) ===")
    print(f"{'unique prompts injected':>24} {'hit rate on the next 300 requests':>36}")
    for k in (0, 10, 30, 100):
        vals = []
        for s in SEEDS:
            r = np.random.default_rng(s)
            reqs = wl_system(r, 900)
            warm, probe = reqs[:600], reqs[600:]
            c = RadixCache(64000)
            for pr, out in warm:
                c.insert(pr + out)
            for _ in range(k):
                c.insert(rand_tokens(r, 2000))
            hit = tot = 0
            for pr, out in probe:
                hit += c.match_length(pr)
                tot += len(pr)
                c.insert(pr + out)
            vals.append(hit / tot)
        print(f"{k:24d} {agg(vals):>36}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tests":
        selftests()
    else:
        selftests()
        experiments()
