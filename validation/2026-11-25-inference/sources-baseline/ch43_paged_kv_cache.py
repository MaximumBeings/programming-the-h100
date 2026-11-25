#!/usr/bin/env python3
"""Chapter 43 - A paged KV cache from scratch (CPU, NumPy only).

Run:  python3 ch43_paged_kv_cache.py          # self-tests + experiments
      python3 ch43_paged_kv_cache.py tests    # self-tests only
"""
import math
import sys
import time
from collections import deque

import numpy as np


class OutOfBlocks(Exception):
    pass


# --------------------------------------------------------------------------
# Block allocator with reference counts
# --------------------------------------------------------------------------
class BlockAllocator:
    def __init__(self, n_blocks):
        self.n = n_blocks
        self.free = deque(range(n_blocks))
        self.ref = [0] * n_blocks

    def num_free(self):
        return len(self.free)

    def alloc(self):
        if not self.free:
            raise OutOfBlocks("no free blocks")
        b = self.free.popleft()
        assert self.ref[b] == 0
        self.ref[b] = 1
        return b

    def incref(self, b):
        if self.ref[b] <= 0:
            raise ValueError(f"incref on free block {b}")
        self.ref[b] += 1

    def decref(self, b):
        if self.ref[b] <= 0:
            raise ValueError(f"double free of block {b}")
        self.ref[b] -= 1
        if self.ref[b] == 0:
            self.free.append(b)


# --------------------------------------------------------------------------
# Paged KV cache: one K and one V pool, per-sequence block tables
# --------------------------------------------------------------------------
class PagedKVCache:
    def __init__(self, n_blocks, block_size, heads, dim, dtype=np.float32):
        self.B, self.H, self.D = block_size, heads, dim
        self.alloc = BlockAllocator(n_blocks)
        self.K = np.zeros((n_blocks, block_size, heads, dim), dtype)
        self.V = np.zeros((n_blocks, block_size, heads, dim), dtype)
        self.seqs = {}  # sid -> {"table": [block ids], "len": int}

    # ---- sequence lifecycle ----
    def add_seq(self, sid):
        if sid in self.seqs:
            raise ValueError(f"sequence {sid} exists")
        self.seqs[sid] = {"table": [], "len": 0}

    def free_seq(self, sid):
        for b in self.seqs.pop(sid)["table"]:
            self.alloc.decref(b)

    def fork(self, parent, child):
        """Share every block of `parent` with a new sequence (no copying)."""
        if child in self.seqs:
            raise ValueError(f"sequence {child} exists")
        p = self.seqs[parent]
        for b in p["table"]:
            self.alloc.incref(b)
        self.seqs[child] = {"table": list(p["table"]), "len": p["len"]}

    # ---- appending ----
    def blocks_needed_to_append(self, sid):
        """Fresh blocks the next append will allocate: 0 or 1."""
        s = self.seqs[sid]
        if s["len"] % self.B == 0:
            return 1  # need a new block
        last = s["table"][-1]
        return 1 if self.alloc.ref[last] > 1 else 0  # copy-on-write

    def append(self, sid, k, v):
        """Append one token's K/V (shape (H, D)). Atomic: on OutOfBlocks nothing changes."""
        s = self.seqs[sid]
        pos = s["len"] % self.B
        if pos == 0:
            b = self.alloc.alloc()  # may raise before any mutation
            s["table"].append(b)
        else:
            last = s["table"][-1]
            if self.alloc.ref[last] > 1:  # shared: copy-on-write
                nb = self.alloc.alloc()  # may raise before any mutation
                self.K[nb] = self.K[last]
                self.V[nb] = self.V[last]
                self.alloc.decref(last)
                s["table"][-1] = nb
            b = s["table"][-1]
        self.K[b, pos] = k
        self.V[b, pos] = v
        s["len"] += 1

    def gather(self, sid):
        """Dense (len, H, D) copies of K and V (reference/debug path)."""
        s = self.seqs[sid]
        if not s["table"]:
            e = np.zeros((0, self.H, self.D), self.K.dtype)
            return e, e.copy()
        K = self.K[s["table"]].reshape(-1, self.H, self.D)[: s["len"]]
        V = self.V[s["table"]].reshape(-1, self.H, self.D)[: s["len"]]
        return K, V

    # ---- accounting ----
    def used_blocks(self):
        return self.alloc.n - self.alloc.num_free()

    def check_invariants(self):
        counts = [0] * self.alloc.n
        for s in self.seqs.values():
            assert len(s["table"]) == math.ceil(s["len"] / self.B)
            for b in s["table"]:
                counts[b] += 1
        assert counts == self.alloc.ref, "refcounts disagree with block tables"
        free = set(self.alloc.free)
        assert len(free) == len(self.alloc.free), "duplicate in free list"
        for b in range(self.alloc.n):
            assert (b in free) == (self.alloc.ref[b] == 0)


# --------------------------------------------------------------------------
# Attention: dense reference, and paged (block-by-block, online softmax)
# --------------------------------------------------------------------------
def dense_attention(q, K, V):
    """q (H,D); K,V (T,H,D) -> (H,D). float64 reference."""
    q, K, V = q.astype(np.float64), K.astype(np.float64), V.astype(np.float64)
    s = np.einsum("hd,thd->ht", q, K) / math.sqrt(q.shape[-1])
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    return np.einsum("ht,thd->hd", p, V)


def paged_attention(q, cache, sid):
    """Walk the block table; never materialises a contiguous K/V (online softmax)."""
    s = cache.seqs[sid]
    q = q.astype(np.float64)
    H, D = q.shape
    m = np.full(H, -np.inf)
    l = np.zeros(H)
    acc = np.zeros((H, D))
    remaining = s["len"]
    for b in s["table"]:
        n = min(cache.B, remaining)
        remaining -= n
        Kb = cache.K[b, :n].astype(np.float64)
        Vb = cache.V[b, :n].astype(np.float64)
        sc = np.einsum("hd,thd->ht", q, Kb) / math.sqrt(D)
        m_new = np.maximum(m, sc.max(axis=1))
        scale = np.exp(m - m_new)
        p = np.exp(sc - m_new[:, None])
        l = l * scale + p.sum(axis=1)
        acc = acc * scale[:, None] + np.einsum("ht,thd->hd", p, Vb)
        m = m_new
    return acc / l[:, None]


# --------------------------------------------------------------------------
# A contiguous (first-fit) allocator, for the external-fragmentation contrast
# --------------------------------------------------------------------------
class ContiguousAllocator:
    def __init__(self, total_slots):
        self.total = total_slots
        self.regions = []  # sorted list of (start, size, sid)

    def free_slots(self):
        return self.total - sum(r[1] for r in self.regions)

    def alloc(self, sid, size):
        pos = 0
        for i, (st, sz, _) in enumerate(self.regions):
            if st - pos >= size:
                self.regions.insert(i, (pos, size, sid))
                return True
            pos = st + sz
        if self.total - pos >= size:
            self.regions.append((pos, size, sid))
            return True
        return False

    def free(self, sid):
        self.regions = [r for r in self.regions if r[2] != sid]


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------
def rand_kv(rng, H, D):
    return rng.standard_normal((H, D)).astype(np.float32), rng.standard_normal((H, D)).astype(np.float32)


def selftests():
    t0 = time.time()
    n_ok = 0

    def ok(name):
        nonlocal n_ok
        n_ok += 1
        print(f"  PASS {n_ok:2d}  {name}")

    rng = np.random.default_rng(0)
    H, D = 2, 8

    # 1 allocator: alloc/free round trip
    a = BlockAllocator(4)
    bs = [a.alloc() for _ in range(4)]
    assert a.num_free() == 0 and sorted(bs) == [0, 1, 2, 3]
    for b in bs:
        a.decref(b)
    assert a.num_free() == 4
    ok("allocator hands out each block once and takes all back")

    # 2 double free and incref-on-free are rejected
    a = BlockAllocator(2)
    b = a.alloc()
    a.decref(b)
    for bad in (lambda: a.decref(b), lambda: a.incref(b)):
        try:
            bad()
            raise AssertionError("not rejected")
        except ValueError:
            pass
    ok("double free and incref of a free block raise")

    # 3 OOM raises and does not corrupt
    a = BlockAllocator(1)
    a.alloc()
    try:
        a.alloc()
        raise AssertionError
    except OutOfBlocks:
        pass
    assert a.num_free() == 0
    ok("allocation fails cleanly when empty")

    # 4 append / gather round trip across block boundaries
    for B in (1, 3, 4, 16):
        c = PagedKVCache(64, B, H, D)
        c.add_seq("s")
        ks, vs = [], []
        for _ in range(23):
            k, v = rand_kv(rng, H, D)
            c.append("s", k, v)
            ks.append(k)
            vs.append(v)
        K, V = c.gather("s")
        assert np.array_equal(K, np.stack(ks)) and np.array_equal(V, np.stack(vs))
        assert len(c.seqs["s"]["table"]) == math.ceil(23 / B)
        c.check_invariants()
    ok("gather returns exactly what was appended (B = 1, 3, 4, 16)")

    # 5 free returns all blocks
    c = PagedKVCache(16, 4, H, D)
    c.add_seq("s")
    for _ in range(10):
        c.append("s", *rand_kv(rng, H, D))
    assert c.used_blocks() == 3
    c.free_seq("s")
    assert c.used_blocks() == 0
    c.check_invariants()
    ok("freeing a sequence returns every block")

    # 6 fork shares blocks without copying
    c = PagedKVCache(16, 4, H, D)
    c.add_seq("p")
    for _ in range(10):
        c.append("p", *rand_kv(rng, H, D))
    used = c.used_blocks()
    c.fork("p", "c")
    assert c.used_blocks() == used
    assert c.seqs["p"]["table"] == c.seqs["c"]["table"]
    c.check_invariants()
    ok("fork allocates nothing; both tables point at the same blocks")

    # 7 copy-on-write: child append does not change the parent
    Kp, Vp = c.gather("p")
    c.append("c", *rand_kv(rng, H, D))  # len 10 -> partial last block is shared -> CoW
    Kp2, Vp2 = c.gather("p")
    assert np.array_equal(Kp, Kp2) and np.array_equal(Vp, Vp2)
    assert c.seqs["p"]["table"][-1] != c.seqs["c"]["table"][-1]
    assert c.seqs["p"]["table"][:-1] == c.seqs["c"]["table"][:-1]
    assert c.used_blocks() == used + 1
    c.check_invariants()
    ok("child append copies the shared partial block; parent unchanged")

    # 8 parent append after fork also copies (symmetry)
    c = PagedKVCache(16, 4, H, D)
    c.add_seq("p")
    for _ in range(6):
        c.append("p", *rand_kv(rng, H, D))
    c.fork("p", "c")
    Kc, _ = c.gather("c")
    c.append("p", *rand_kv(rng, H, D))
    Kc2, _ = c.gather("c")
    assert np.array_equal(Kc, Kc2)
    c.check_invariants()
    ok("parent append after fork leaves the child unchanged")

    # 9 fork at a block boundary needs no copy
    c = PagedKVCache(16, 4, H, D)
    c.add_seq("p")
    for _ in range(8):
        c.append("p", *rand_kv(rng, H, D))
    c.fork("p", "c")
    assert c.blocks_needed_to_append("c") == 1  # fresh block, not a copy
    before = c.used_blocks()
    c.append("c", *rand_kv(rng, H, D))
    assert c.used_blocks() == before + 1
    assert c.seqs["p"]["table"] == c.seqs["c"]["table"][:-1]
    ok("fork at a block boundary shares all blocks; append adds one fresh block")

    # 10 last owner writes in place (no copy once the refcount is 1)
    c = PagedKVCache(16, 4, H, D)
    c.add_seq("p")
    for _ in range(6):
        c.append("p", *rand_kv(rng, H, D))
    c.fork("p", "c")
    c.free_seq("p")
    before = c.used_blocks()
    assert c.blocks_needed_to_append("c") == 0
    c.append("c", *rand_kv(rng, H, D))
    assert c.used_blocks() == before
    c.check_invariants()
    ok("after the parent is freed the child appends in place")

    # 11 OOM during append leaves state untouched (atomic)
    c = PagedKVCache(2, 4, H, D)
    c.add_seq("s")
    for _ in range(8):
        c.append("s", *rand_kv(rng, H, D))
    snap = (list(c.seqs["s"]["table"]), c.seqs["s"]["len"], list(c.alloc.ref))
    try:
        c.append("s", *rand_kv(rng, H, D))
        raise AssertionError
    except OutOfBlocks:
        pass
    assert snap == (c.seqs["s"]["table"], c.seqs["s"]["len"], c.alloc.ref)
    c.check_invariants()
    ok("append that hits OutOfBlocks changes nothing")

    # 12 OOM during copy-on-write is also atomic
    c = PagedKVCache(2, 4, H, D)
    c.add_seq("p")
    for _ in range(6):
        c.append("p", *rand_kv(rng, H, D))  # 2 blocks, pool full
    c.fork("p", "c")
    snap = (list(c.seqs["c"]["table"]), c.seqs["c"]["len"], list(c.alloc.ref))
    try:
        c.append("c", *rand_kv(rng, H, D))
        raise AssertionError
    except OutOfBlocks:
        pass
    assert snap == (c.seqs["c"]["table"], c.seqs["c"]["len"], c.alloc.ref)
    c.check_invariants()
    ok("copy-on-write that hits OutOfBlocks changes nothing")

    # 13 paged attention == dense attention (float64 reference)
    worst = 0.0
    for B in (1, 2, 5, 16):
        for T in (1, 2, 7, 33, 100):
            c = PagedKVCache(256, B, H, D)
            c.add_seq("s")
            for _ in range(T):
                c.append("s", *rand_kv(rng, H, D))
            q = rng.standard_normal((H, D)).astype(np.float32)
            K, V = c.gather("s")
            worst = max(worst, float(np.abs(paged_attention(q, c, "s") - dense_attention(q, K, V)).max()))
    assert worst < 1e-9, worst
    ok(f"paged attention matches dense attention (max abs diff {worst:.1e})")

    # 14 attention is independent of physical block placement
    c = PagedKVCache(64, 4, H, D)
    c.add_seq("a")
    c.add_seq("b")
    toks = [rand_kv(rng, H, D) for _ in range(30)]
    for k, v in toks:  # interleave so a's blocks are scattered
        c.append("a", k, v)
        c.append("b", *rand_kv(rng, H, D))
    q = rng.standard_normal((H, D)).astype(np.float32)
    K, V = c.gather("a")
    assert not all(x + 1 == y for x, y in zip(c.seqs["a"]["table"], c.seqs["a"]["table"][1:]))
    assert np.abs(paged_attention(q, c, "a") - dense_attention(q, K, V)).max() < 1e-9
    ok("attention is correct when a sequence's blocks are not contiguous")

    # 15 forked sequences attend identically until they diverge
    c = PagedKVCache(64, 4, H, D)
    c.add_seq("p")
    for _ in range(11):
        c.append("p", *rand_kv(rng, H, D))
    c.fork("p", "c")
    q = rng.standard_normal((H, D)).astype(np.float32)
    assert np.array_equal(paged_attention(q, c, "p"), paged_attention(q, c, "c"))
    ok("fork attends identically to its parent before any divergence")

    # 16 property test: random ops against a plain-list reference model
    for seed in range(20):
        r = np.random.default_rng(seed)
        B = int(r.choice([1, 2, 4, 8]))
        c = PagedKVCache(48, B, 1, 2)
        ref = {}
        nid = 0
        for _ in range(400):
            op = r.choice(["add", "append", "append", "append", "fork", "free"])
            ids = list(ref)
            if op == "add" and len(ids) < 8:
                c.add_seq(nid)
                ref[nid] = []
                nid += 1
            elif op == "append" and ids:
                sid = ids[int(r.integers(len(ids)))]
                k, v = rand_kv(r, 1, 2)
                try:
                    c.append(sid, k, v)
                    ref[sid].append((k, v))
                except OutOfBlocks:
                    pass
            elif op == "fork" and ids and len(ids) < 8:
                sid = ids[int(r.integers(len(ids)))]
                c.fork(sid, nid)
                ref[nid] = list(ref[sid])
                nid += 1
            elif op == "free" and ids:
                sid = ids[int(r.integers(len(ids)))]
                c.free_seq(sid)
                del ref[sid]
            c.check_invariants()
            for sid, toks_ in ref.items():
                K, V = c.gather(sid)
                assert K.shape[0] == len(toks_)
                if toks_:
                    assert np.array_equal(K, np.stack([t[0] for t in toks_]))
                    assert np.array_equal(V, np.stack([t[1] for t in toks_]))
        for sid in list(ref):
            c.free_seq(sid)
        assert c.used_blocks() == 0
    ok("20 random sessions (add/append/fork/free, with OOM) match a reference and never leak")

    # 17 contiguous allocator sanity
    ca = ContiguousAllocator(10)
    assert ca.alloc("a", 4) and ca.alloc("b", 4) and not ca.alloc("c", 4)
    ca.free("a")
    assert ca.alloc("c", 4) and ca.free_slots() == 2
    ok("first-fit contiguous allocator reuses freed holes")

    print(f"ALL PASS   ({time.time() - t0:.0f} s)  [{n_ok} tests]")


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def sample_lengths(rng, n, max_len=2048, median=300, sigma=0.9):
    x = np.exp(rng.normal(math.log(median), sigma, n))
    return np.clip(np.rint(x), 1, max_len).astype(int)


def experiments():
    rng = np.random.default_rng(1)
    MAXLEN = 2048
    SLOTS = 65536  # KV budget in token slots

    print("\n=== Experiment 1: internal waste versus block size ===")
    lens = sample_lengths(rng, 200000, MAXLEN)
    print(f"lengths: lognormal median 300, clipped to [1, {MAXLEN}]; mean {lens.mean():.1f}")
    print(f"{'B':>4} {'waste/seq (slots)':>18} {'theory (B-1)/2':>15} {'waste % of live':>16} {'table entries/seq':>18}")
    for B in (1, 4, 8, 16, 32, 64, 128, 256):
        alloc = np.ceil(lens / B) * B
        w = (alloc - lens).mean()
        print(f"{B:4d} {w:18.2f} {(B - 1) / 2:15.2f} {100 * w / lens.mean():15.2f}% {np.ceil(lens / B).mean():18.1f}")

    print(f"\n=== Experiment 2: how many sequences fit in {SLOTS} token slots ===")
    print(f"contiguous, reserve max_len={MAXLEN} per sequence: {SLOTS // MAXLEN} sequences")
    for B in (1, 16, 64, 256):
        r = np.random.default_rng(2)
        counts = []
        for _ in range(200):
            used, n = 0, 0
            while True:
                L = int(sample_lengths(r, 1, MAXLEN)[0])
                need = math.ceil(L / B) * B
                if used + need > SLOTS:
                    break
                used += need
                n += 1
            counts.append(n)
        print(f"paged B={B:3d}: {np.mean(counts):7.1f} sequences  ({np.mean(counts) / (SLOTS // MAXLEN):5.1f}x)")

    print("\n=== Experiment 3: prefix sharing by fork (prompt P, n children, g new tokens each) ===")
    print(f"{'B':>3} {'P':>5} {'n':>3} {'g':>4} {'blocks (fork)':>14} {'formula':>8} {'blocks (copy)':>14} {'saving':>7}")
    H, D = 1, 2
    for B, P, n, g in [(16, 512, 8, 64), (16, 520, 8, 64), (16, 1000, 16, 128), (64, 1000, 16, 128), (16, 100, 4, 400)]:
        c = PagedKVCache(4096, B, H, D)
        c.add_seq("p")
        for _ in range(P):
            c.append("p", np.zeros((H, D), np.float32), np.zeros((H, D), np.float32))
        for i in range(n):
            c.fork("p", i)
        c.free_seq("p")
        for _ in range(g):
            for i in range(n):
                c.append(i, np.ones((H, D), np.float32), np.ones((H, D), np.float32))
        got = c.used_blocks()
        formula = P // B + n * math.ceil((P % B + g) / B)
        copy = n * math.ceil((P + g) / B)
        assert got == formula, (got, formula)
        c.check_invariants()
        print(f"{B:3d} {P:5d} {n:3d} {g:4d} {got:14d} {formula:8d} {copy:14d} {100 * (1 - got / copy):6.1f}%")

    print("\n=== Experiment 4: external fragmentation, contiguous versus paged (saturated) ===")
    # Arrivals (1 per step) outpace departures (0.5 per step), so both systems fill up.
    # We compare how much of the pool holds live tokens, and how many refusals happened
    # although the TOTAL free space was big enough ("refused despite space").
    B = 16
    r = np.random.default_rng(3)
    ca = ContiguousAllocator(SLOTS)
    paged_free = SLOTS // B
    live_c, live_p = {}, {}
    ref_c = ref_p = despite_c = despite_p = 0
    util_c, util_p = [], []
    steps = 20000
    for step in range(steps):
        if live_c and r.random() < 0.5:
            sid = list(live_c)[int(r.integers(len(live_c)))]
            ca.free(sid)
            del live_c[sid]
        if live_p and r.random() < 0.5:
            sid = list(live_p)[int(r.integers(len(live_p)))]
            paged_free += live_p.pop(sid)[0]
        L = int(sample_lengths(r, 1, MAXLEN, median=200)[0])
        if not ca.alloc(step, L):
            ref_c += 1
            despite_c += ca.free_slots() >= L
        else:
            live_c[step] = L
        need = math.ceil(L / B)
        if need > paged_free:
            ref_p += 1
            despite_p += paged_free * B >= L
        else:
            paged_free -= need
            live_p[step] = (need, L)
        if step >= steps // 4:
            util_c.append(sum(live_c.values()) / SLOTS)
            util_p.append(sum(v[1] for v in live_p.values()) / SLOTS)
    print(f"{'':>11} {'refused':>8} {'refused despite enough total free':>34} {'mean live tokens / pool':>24}")
    print(f"{'contiguous':>11} {ref_c:8d} {despite_c:34d} {np.mean(util_c):24.3f}")
    print(f"{'paged B=16':>11} {ref_p:8d} {despite_p:34d} {np.mean(util_p):24.3f}")

    print("\n=== KV bytes per token (hand arithmetic) ===")
    for name, layers, kvh, hd in [("Llama-3-8B (32 layers, 8 KV heads, 128)", 32, 8, 128),
                                  ("Llama-3-70B (80 layers, 8 KV heads, 128)", 80, 8, 128)]:
        per_tok = 2 * layers * kvh * hd * 2
        print(f"{name}: {per_tok} B/token = {per_tok / 1024:.0f} KiB; "
              f"{MAXLEN}-token reservation = {per_tok * MAXLEN / 2**20:.0f} MiB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tests":
        selftests()
    else:
        selftests()
        experiments()
