#!/usr/bin/env python3
"""Chapter 38: a semantic response cache, its threshold, its false hits, its economics, and its failure modes.

Run:  python3 ch38_semantic_cache.py            # self-tests + all experiments (about 30-60 s, CPU, needs numpy)
      python3 ch38_semantic_cache.py --selftest # self-tests only

WHAT IS REAL: the cache data structure (partitioned by tenant/model/params, TTL, LRU, cosine lookup), the Wilson
confidence interval, the economics formulas, and the experimental method.
WHAT IS INVENTED: the "embeddings". There is no embedding model here: queries are synthetic unit vectors in which a
paraphrase is its intent's centre plus noise, and "sibling" intents (the same question with a different number,
negation, entity...) are centres placed close together. Every similarity value, every threshold and every rate printed
below is a property of this synthetic world. NONE of them transfers to a real embedding model; Section 38.9 shows how
to measure the real ones.
"""
import math
import sys
import time

import numpy as np


# =============================================================== 1. synthetic world (stand-in for an embedding model)
def unit(m):
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


class World:
    """F families of M sibling intents. Siblings are near each other (hard negatives); families are far apart.
    s_para: paraphrase noise (bigger = noisier embedding model). s_sib: sibling spread inside a family."""
    def __init__(self, seed=0, families=60, members=4, dim=64, s_para=0.20, s_sib=0.30):
        self.rng = np.random.default_rng(seed)
        self.dim, self.F, self.M, self.s_para = dim, families, members, s_para
        fam = unit(self.rng.standard_normal((families, dim)))
        self.family = np.repeat(np.arange(families), members)
        self.centres = unit(fam[self.family] + s_sib * self._noise(families * members))
        self.n_intents = families * members

    def _noise(self, rows):
        return self.rng.standard_normal((rows, self.dim)) / math.sqrt(self.dim)

    def paraphrase(self, intents):
        intents = np.asarray(intents)
        return unit(self.centres[intents] + self.s_para * self._noise(len(intents)))

    def stream(self, n, zipf=1.0, dup_frac=0.10):
        """n queries: intents ~ Zipf; a fraction are exact repeats of an earlier query vector."""
        p = 1.0 / np.arange(1, self.n_intents + 1) ** zipf
        p /= p.sum()
        intents = self.rng.choice(self.n_intents, size=n, p=p)
        vecs = self.paraphrase(intents)
        for i in range(1, n):
            if self.rng.random() < dup_frac:                      # exact repeat of an earlier request
                j = self.rng.integers(0, i)
                intents[i], vecs[i] = intents[j], vecs[j]
        return intents, vecs


# =============================================================== 2. the cache
class SemanticCache:
    """Cosine-similarity response cache.

    Entries live in PARTITIONS keyed by whatever must be identical for an answer to be reusable (tenant, model,
    system prompt hash, sampling parameters...). A lookup only ever compares against its own partition.
    Per partition: at most `max_entries` (LRU eviction) and entries older than `ttl` seconds are invalid (strict:
    an entry created at t is valid for now < t + ttl). Vectors must be unit length; lookup returns the best match
    at or above the threshold, or None."""
    def __init__(self, dim, threshold, ttl=math.inf, max_entries=100000):
        if dim <= 0 or max_entries <= 0 or ttl <= 0 or math.isnan(ttl):
            raise ValueError("dimension, capacity and TTL must be positive")
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("cosine threshold must be finite and in [-1, 1]")
        self.dim, self.threshold, self.ttl, self.max_entries = dim, threshold, ttl, max_entries
        self.parts = {}                                           # key -> dict(mat, n, meta[list of dict])

    def _part(self, key):
        if key not in self.parts:
            self.parts[key] = dict(mat=np.zeros((64, self.dim)), n=0, meta=[])
        return self.parts[key]

    def _vector(self, vec):
        vec = np.asarray(vec, dtype=float)
        if vec.shape != (self.dim,) or not np.isfinite(vec).all():
            raise ValueError("cache vector must have the declared dimension and finite values")
        if not np.isclose(np.linalg.norm(vec), 1.0, rtol=1e-5, atol=1e-7):
            raise ValueError("cosine cache requires unit-length vectors")
        return vec

    def lookup(self, key, vec, now=0.0):
        vec = self._vector(vec)
        p = self.parts.get(key)
        if p is None or p["n"] == 0:
            return None, -1.0
        sims = p["mat"][:p["n"]] @ vec
        created = np.array([m["created"] for m in p["meta"]])
        sims = np.where(now < created + self.ttl, sims, -np.inf)    # expired entries cannot match
        j = int(np.argmax(sims))
        if sims[j] >= self.threshold:
            p["meta"][j]["last_used"] = now
            return p["meta"][j], float(sims[j])
        return None, float(sims[j]) if np.isfinite(sims[j]) else -1.0

    def insert(self, key, vec, answer, now=0.0, **extra):
        vec = self._vector(vec)
        p = self._part(key)
        meta = dict(answer=answer, created=now, last_used=now, **extra)
        if p["n"] < self.max_entries:
            if p["n"] == p["mat"].shape[0]:
                p["mat"] = np.vstack([p["mat"], np.zeros_like(p["mat"])])
            slot = p["n"]; p["n"] += 1
            p["meta"].append(meta)
        else:                                                      # evict least recently used (ties: oldest created)
            slot = min(range(p["n"]), key=lambda i: (p["meta"][i]["last_used"], p["meta"][i]["created"]))
            p["meta"][slot] = meta
        p["mat"][slot] = vec
        return meta

    def size(self, key=None):
        return sum(p["n"] for p in self.parts.values()) if key is None else self.parts.get(key, {"n": 0})["n"]


# =============================================================== 3. statistics and economics
def wilson(x, n, z=1.959964):
    """95% Wilson score interval for a binomial proportion. Well behaved at x=0 and for small p, unlike the normal interval."""
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def wilson_exact_coverage(n, p, z=1.959964):
    """EXACT coverage of the Wilson interval for Binomial(n, p): sum the pmf over every x whose interval contains p."""
    cov = 0.0
    for x in range(n + 1):
        lo, hi = wilson(x, n, z)
        if lo <= p <= hi:
            cov += math.comb(n, x) * p ** x * (1 - p) ** (n - x)
    return cov


def cluster_bootstrap_ratio(num, den, B=2000, seed=0):
    """95% percentile bootstrap CI for sum(num)/sum(den) resampling CLUSTERS (here: cache entries), because the
    false hits of one bad entry are not independent events. Returns None if there are no events at all."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if num.sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (B, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def exact_match_stream(vecs):
    """Baseline: a hash (exact-match) cache. Hit iff the identical vector was seen before. Zero false hits by construction
    (identical vector = identical intent here). This, not 'no cache', is what a semantic cache must beat."""
    seen, hits = set(), 0
    for v in vecs:
        k = v.tobytes()
        if k in seen:
            hits += 1
        else:
            seen.add(k)
    return hits / len(vecs)


def cost_per_request(hit_rate, wrong_rate, c_llm, c_emb, c_bad):
    """Every request pays the embedding/lookup; misses pay the LLM; wrong answers pay c_bad (escalation, refund, trust)."""
    return c_emb + (1 - hit_rate) * c_llm + wrong_rate * c_bad


def breakeven_bad_cost(hit_rate, wrong_rate, c_llm, c_emb):
    """Largest c_bad at which the cache is still no worse than no cache: c_llm = cost_per_request(...)."""
    net_saved = hit_rate * c_llm - c_emb
    if wrong_rate > 0:
        return net_saved / wrong_rate
    # With no wrong answers c_bad is irrelevant, but a cache with insufficient
    # hits may still fail to pay for its embedding cost even when c_bad=0.
    return math.inf if net_saved >= 0 else -math.inf


# =============================================================== 4. experiments
def run_stream(world, intents, vecs, threshold, ttl=math.inf, key_fn=None, answer_fn=None, now_fn=None, versions=None):
    """Replay a stream. Miss -> 'call the model' (the correct answer for that intent) and insert. Hit -> serve the cached
    answer. Returns counts and per-request details."""
    cache = SemanticCache(world.dim, threshold, ttl)
    n = len(intents)
    hits = false_hits = stale = 0
    entries = []                                                   # every inserted entry; hits are CLUSTERED on entries
    for i in range(n):
        now = now_fn(i) if now_fn else float(i)
        key = key_fn(i) if key_fn else "default"
        truth = answer_fn(i) if answer_fn else int(intents[i])
        entry, sim = cache.lookup(key, vecs[i], now)
        if entry is not None:
            hits += 1
            entry["served"] = entry.get("served", 0) + 1
            if entry["answer"] != truth:
                false_hits += 1
                entry["bad"] = entry.get("bad", 0) + 1
        else:
            entries.append(cache.insert(key, vecs[i], truth, now))
    served = np.array([e.get("served", 0) for e in entries]); bad = np.array([e.get("bad", 0) for e in entries])
    return dict(n=n, hits=hits, false_hits=false_hits, served=served, bad=bad)


def cosine_table(world, n_pairs=20000):
    rng = np.random.default_rng(99)
    a = rng.integers(0, world.n_intents, n_pairs)
    same = (world.paraphrase(a) * world.paraphrase(a)).sum(1)
    sib_b = (a // world.M) * world.M + (a % world.M + rng.integers(1, world.M, n_pairs)) % world.M
    sib = (world.paraphrase(a) * world.paraphrase(sib_b)).sum(1)
    other_b = (a + world.M * rng.integers(1, world.F, n_pairs)) % world.n_intents
    other = (world.paraphrase(a) * world.paraphrase(other_b)).sum(1)
    return same, sib, other


def threshold_sweep(world, taus, n=4000, seeds=(1, 2, 3)):
    out = {}
    for tau in taus:
        h = f = t = 0
        sv, bd = [], []
        for s in seeds:
            w = World(seed=s, s_para=world.s_para)
            intents, vecs = w.stream(n)
            r = run_stream(w, intents, vecs, tau)
            h += r["hits"]; f += r["false_hits"]; t += r["n"]
            sv.append(r["served"]); bd.append(r["bad"])
        out[tau] = dict(n=t, hits=h, false_hits=f, served=np.concatenate(sv), bad=np.concatenate(bd))
    return out


def selftests():
    res = []

    def check(name, cond):
        res.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    w = World(seed=0)
    same, sib, other = cosine_table(w, 8000)
    check(f"world sanity: mean cosine same-intent ({same.mean():.3f}) > sibling ({sib.mean():.3f}) > unrelated ({other.mean():.3f})",
          same.mean() > sib.mean() > other.mean())
    check("all synthetic vectors have unit norm", np.allclose(np.linalg.norm(w.paraphrase(np.arange(50)), axis=1), 1.0))

    # exact duplicate hits at (almost) threshold 1
    c = SemanticCache(w.dim, 0.999999)
    v = w.paraphrase([3])[0]
    c.insert("t", v, "ans")
    e, s = c.lookup("t", v)
    check("an exact repeat of a stored vector hits at threshold 0.999999 with similarity 1", e is not None and abs(s - 1.0) < 1e-9)

    # partition isolation (property test over random operations)
    rng = np.random.default_rng(5)
    c = SemanticCache(w.dim, 0.5)
    ok, nhit = True, 0
    for step in range(600):
        key = f"tenant{rng.integers(0, 3)}"
        vec = w.paraphrase([int(rng.integers(0, w.n_intents))])[0]
        if rng.random() < 0.5:
            c.insert(key, vec, ("owner", key))
        else:
            e, _ = c.lookup(key, vec)
            nhit += e is not None
            ok &= (e is None or e["answer"] == ("owner", key))
    check(f"partition isolation: in 600 random operations ({nhit} of them hits, so the test exercises hits) no lookup returns another partition's entry. By construction of the dict keying; this guards regressions", ok and nhit > 0)

    # TTL, strict boundary
    c = SemanticCache(w.dim, 0.9, ttl=100.0)
    c.insert("k", v, "a", now=0.0)
    check("TTL: valid at t=99.9, expired at exactly t=100 (strict) and after",
          c.lookup("k", v, 99.9)[0] is not None and c.lookup("k", v, 100.0)[0] is None and c.lookup("k", v, 500.0)[0] is None)

    # LRU eviction
    c = SemanticCache(w.dim, 0.99, max_entries=3)
    vs = w.paraphrase([0, 10, 20, 30])
    for i in range(3):
        c.insert("k", vs[i], i, now=float(i))
    c.lookup("k", vs[0], now=10.0)                                  # touch entry 0 so entry 1 is now the LRU
    c.insert("k", vs[3], 3, now=11.0)
    kept = {m["answer"] for m in c.parts["k"]["meta"]}
    check("LRU: capacity 3, after touching entry 0 the least recently used (entry 1) is evicted", kept == {0, 2, 3})

    # determinism
    r1 = run_stream(*(lambda ww: (ww, *ww.stream(1500)))(World(seed=4)), 0.95)
    r2 = run_stream(*(lambda ww: (ww, *ww.stream(1500)))(World(seed=4)), 0.95)
    check("replaying the same seeded stream gives identical counts and per-entry tallies",
          (r1["n"], r1["hits"], r1["false_hits"]) == (r2["n"], r2["hits"], r2["false_hits"]) and np.array_equal(r1["served"], r2["served"]) and np.array_equal(r1["bad"], r2["bad"]))

    # monotonicity in the threshold
    taus = (0.80, 0.88, 0.92, 0.95, 0.97, 0.99)
    sw = threshold_sweep(w, taus, n=2500, seeds=(1, 2))
    hr = [sw[t]["hits"] / sw[t]["n"] for t in taus]
    check(f"EMPIRICAL (not a theorem for a streaming cache: a higher threshold means more inserts, which can mean more later hits): hit rate is nonincreasing over this threshold grid {[round(x, 2) for x in hr]}", all(a >= b - 1e-9 for a, b in zip(hr, hr[1:])))
    fr = {t: sw[t]["false_hits"] / max(1, sw[t]["hits"]) for t in taus}
    check(f"false-hit rate at threshold 0.99 ({fr[0.99]:.3f}) is below that at 0.80 ({fr[0.80]:.3f})", fr[0.99] < fr[0.80])

    # Wilson interval
    lo, hi = wilson(0, 200)
    check(f"Wilson upper bound for 0 events in 200 trials is {hi:.4f} (about 3/n = 0.015 to 0.019)", 0.018 < hi < 0.0195 and lo == 0.0)
    cov = wilson_exact_coverage(200, 0.02)
    check(f"Wilson 95% interval, EXACT coverage at n=200, p=0.02 (binomial sum, no simulation): {cov:.4f}; asserted in [0.92, 0.98] (Wilson coverage oscillates with n and p)",
          0.92 <= cov <= 0.98)
    check("cluster bootstrap: with 5 entries carrying all 40 false hits of 100 hits the interval is wide, and with no events it returns None",
          cluster_bootstrap_ratio(np.array([8, 8, 8, 8, 8, 0, 0, 0, 0, 0]), np.array([10] * 10)) is not None
          and cluster_bootstrap_ratio(np.zeros(10), np.ones(10)) is None)
    lo, hi = cluster_bootstrap_ratio(np.array([8, 8, 8, 8, 8, 0, 0, 0, 0, 0]), np.array([10] * 10))
    check(f"... and it is wider than the naive Wilson interval on the same 40/100: cluster [{lo:.2f},{hi:.2f}] vs Wilson [{wilson(40, 100)[0]:.2f},{wilson(40, 100)[1]:.2f}]",
          (hi - lo) > (wilson(40, 100)[1] - wilson(40, 100)[0]))
    ex = exact_match_stream(np.vstack([v[None], v[None], w.paraphrase([5]), v[None]]))
    check(f"exact-match baseline: 4 queries (v, v, new, v) -> hit rate {ex:.2f} (= 2/4: the 2nd and 4th are repeats; the 3rd is new)", abs(ex - 0.5) < 1e-12)

    # economics: hand computation
    cpr = cost_per_request(0.40, 0.01, c_llm=0.004, c_emb=0.00002, c_bad=0.10)
    check(f"cost per request by hand: 0.00002 + 0.6*0.004 + 0.01*0.10 = {0.00002 + 0.0024 + 0.001:.5f}; function gives {cpr:.5f}",
          abs(cpr - 0.00342) < 1e-12)
    be = breakeven_bad_cost(0.40, 0.01, 0.004, 0.00002)
    check(f"break-even bad-answer cost = (0.40*0.004 - 0.00002)/0.01 = {be:.4f}, and at that cost the cache equals no cache",
          abs(cost_per_request(0.40, 0.01, 0.004, 0.00002, be) - 0.004) < 1e-12)
    return all(res)


def experiments():
    w = World(seed=0)
    print("\n== 1. Cosine similarity of query pairs in the synthetic world (NOT a real embedding model) ==")
    same, sib, other = cosine_table(w)
    q = lambda a, p: np.percentile(a, p)
    print(f"  {'pair type':28s} {'mean':>6s} {'p1':>6s} {'p5':>6s} {'p50':>6s} {'p95':>6s} {'p99':>6s}")
    for name, a in (("same intent (paraphrase)", same), ("sibling intent (hard negative)", sib), ("unrelated", other)):
        print(f"  {name:28s} {a.mean():6.3f} {q(a,1):6.3f} {q(a,5):6.3f} {q(a,50):6.3f} {q(a,95):6.3f} {q(a,99):6.3f}")
    print("\n  threshold   recall of paraphrases   sibling false-match rate")
    for tau in (0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99):
        print(f"  {tau:9.2f}   {np.mean(same >= tau):21.3f}   {np.mean(sib >= tau):24.3f}")

    print("\n== 2. Streaming cache: hit rate and false hits versus threshold (3 seeds x 4000 queries, Zipf 1.0, 10% exact repeats) ==")
    taus = (0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.999)
    sw = threshold_sweep(w, taus)
    print(f"  {'tau':>6s} {'hit rate':>9s} {'false/hit':>10s} {'Wilson (iid, too narrow)':>26s} {'cluster bootstrap':>20s} {'bad entries':>12s} {'wrong/request':>14s}")
    for t in taus:
        r = sw[t]; fr = r["false_hits"] / max(1, r["hits"]); lo, hi = wilson(r["false_hits"], r["hits"])
        cb = cluster_bootstrap_ratio(r["bad"], r["served"])
        cbs = f"[{cb[0]:7.4f},{cb[1]:7.4f}]" if cb else "(0 events)".rjust(17)
        nserv = int((r["served"] > 0).sum()); nbad = int((r["bad"] > 0).sum())
        print(f"  {t:6.3f} {r['hits']/r['n']:9.3f} {fr:10.4f} [{lo:7.4f},{hi:7.4f}] {cbs:>20s} {nbad:5d}/{nserv:<6d} {r['false_hits']/r['n']:14.4f}")
    r95 = sw[0.95]
    print(f"  at tau=0.95 there were 0 false hits among {int((r95['served'] > 0).sum())} serving entries: Wilson upper bound on the BAD-ENTRY rate = {wilson(0, int((r95['served'] > 0).sum()))[1]:.4f}")
    r90 = sw[0.90]
    print(f"  at tau=0.90 the {r90['false_hits']} false hits came from only {int((r90['bad'] > 0).sum())} distinct entries (clustering)")

    xs = [exact_match_stream(World(seed=s_).stream(4000)[1]) for s_ in (1, 2, 3)]
    exact_hit = float(np.mean(xs))
    print(f"\n  EXACT-MATCH (hash) cache baseline on the same streams: hit rate {exact_hit:.3f}, zero false hits. At tau >= 0.98 the semantic cache's whole hit rate is this.")

    print("\n== 3. Economics (ASSUMED costs): c_llm=$0.004, c_emb=$0.00002 per request; sweep the cost of a wrong answer ==")
    c_llm, c_emb = 0.004, 0.00002
    print(f"  {'tau':>6s}" + "".join(f"  c_bad=${b:<5g}" for b in (0.01, 0.05, 0.25, 1.0)) + "   break-even c_bad")
    best = {}
    for b in (0.01, 0.05, 0.25, 1.0):
        best[b] = max(taus, key=lambda t: c_llm - cost_per_request(sw[t]["hits"] / sw[t]["n"], sw[t]["false_hits"] / sw[t]["n"], c_llm, c_emb, b))
    for t in taus:
        r = sw[t]; h, wr = r["hits"] / r["n"], r["false_hits"] / r["n"]
        cells = "".join(f"  {1e3 * (c_llm - cost_per_request(h, wr, c_llm, c_emb, b)):9.3f}" for b in (0.01, 0.05, 0.25, 1.0))
        be = breakeven_bad_cost(h, wr, c_llm, c_emb)
        print(f"  {t:6.3f}{cells}   {('inf' if be == math.inf else f'{be:8.3f}'):>8s}")
    print("  (cells: net saving in milli-dollars per request versus NO cache; negative = the cache costs money)")
    print(f"  Exact-match baseline saves {1e3 * exact_hit * c_llm:.3f} milli-dollars per request at any c_bad (no false hits). The semantic cache's EXTRA saving over it is the cell minus that.")
    print("  best threshold by c_bad: " + ", ".join(f"${b}: {best[b]}" for b in best))
    h93, w93, h95, w95 = sw[0.93]["hits"] / sw[0.93]["n"], sw[0.93]["false_hits"] / sw[0.93]["n"], sw[0.95]["hits"] / sw[0.95]["n"], sw[0.95]["false_hits"] / sw[0.95]["n"]
    cross = (h93 - h95) * c_llm / (w93 - w95) if w93 > w95 else math.inf
    print(f"  crossover c_bad between tau=0.93 and 0.95: {cross:.3f}  (above it 0.95 wins). tau=0.93 has only {sw[0.93]['false_hits']} false hits in {sw[0.93]['n']} requests;")
    for k in (max(0, sw[0.93]['false_hits'] - 1), sw[0.93]['false_hits'] + 1):
        wk = k / sw[0.93]["n"]
        ck = (h93 - h95) * c_llm / (wk - w95) if wk > w95 else math.inf
        print(f"    with {k} false hits instead the crossover would be {ck:.3f}")

    print("\n== 4. Why '98%' is meaningless without the model: the same threshold on two embedding models ==")
    for name, sp in (("model A (tight paraphrases, s_para=0.20)", 0.20), ("model B (noisier paraphrases, s_para=0.35)", 0.35)):
        ww = World(seed=0, s_para=sp)
        rs = threshold_sweep(ww, (0.95, 0.98), n=3000)
        for t in (0.95, 0.98):
            r = rs[t]
            print(f"  {name:46s} tau={t}: hit rate {r['hits']/r['n']:.3f}  false/hit {r['false_hits']/max(1,r['hits']):.4f}")

    print("\n== 5. Cross-tenant leak: same questions, tenant-specific answers; cache keyed with and without the tenant ==")
    w5 = World(seed=11)
    n = 4000
    intents, vecs = w5.stream(n)
    tenant = np.random.default_rng(3).integers(0, 2, n)
    for label, keyfn in (("one shared partition", lambda i: "all"), ("partition per tenant", lambda i: f"t{tenant[i]}")):
        r = run_stream(w5, intents, vecs, 0.95, key_fn=keyfn, answer_fn=lambda i: (int(tenant[i]), int(intents[i])))
        print(f"  {label:22s}: hits {r['hits']:5d}  answers belonging to the wrong tenant {r['false_hits']:5d} ({r['false_hits']/max(1,r['hits']):.3f} of hits)")

    print("\n== 6. Staleness: tau=0.97; 25% of answers change at t=2000 (1 request/second, horizon 4000 s, so only 2000 s follow the change; TTL 3600 therefore behaves like infinity) ==")
    w6 = World(seed=21)
    n = 4000
    intents, vecs = w6.stream(n, dup_frac=0.0)
    changed = np.random.default_rng(8).random(w6.n_intents) < 0.25
    version = lambda i: 1 if (i >= 2000 and changed[intents[i]]) else 0
    for ttl in (math.inf, 3600.0, 600.0, 120.0):
        cache = SemanticCache(w6.dim, 0.97, ttl)
        hits_after = stale_after = hits = 0
        for i in range(n):
            e, _ = cache.lookup("k", vecs[i], float(i))
            truth = (int(intents[i]), version(i))
            if e is not None:
                hits += 1
                if i >= 2000:
                    hits_after += 1
                    stale_after += e["answer"] != truth
            else:
                cache.insert("k", vecs[i], truth, float(i))
        print(f"  ttl={'inf' if ttl == math.inf else int(ttl):>5}: hit rate {hits / n:.3f}   after the change: {stale_after:4d} stale of {hits_after:4d} hits ({stale_after / max(1, hits_after):.3f})")


def calibrate(sims, same, taus, max_false_hit=0.01):
    """Pick the LOWEST threshold whose Wilson UPPER bound on the false-match rate among DIFFERENT-intent pairs is
    <= max_false_hit. sims: cosine per labelled pair; same: bool per pair (True = same intent / reusable answer).
    Returns (chosen_tau or None, table rows (tau, recall, false_x, false_n, upper))."""
    sims, same = np.asarray(sims), np.asarray(same, bool)
    diff = sims[~same]
    rows, chosen = [], None
    for t in sorted(taus):
        x = int((diff >= t).sum())
        up = wilson(x, len(diff))[1]
        rows.append((t, float((sims[same] >= t).mean()), x, len(diff), up))
        if chosen is None and up <= max_false_hit:
            chosen = t
    return chosen, rows


def calibrate_cli(path, model_name):
    """UNTESTED against a real model. pairs.jsonl lines: {"a": "...", "b": "...", "same": true|false}"""
    import json
    from sentence_transformers import SentenceTransformer          # pip install sentence-transformers
    model = SentenceTransformer(model_name)
    pairs = [json.loads(l) for l in open(path)]
    ea = model.encode([p["a"] for p in pairs], normalize_embeddings=True)
    eb = model.encode([p["b"] for p in pairs], normalize_embeddings=True)
    sims = (ea * eb).sum(1)
    chosen, rows = calibrate(sims, [p["same"] for p in pairs], [round(0.80 + 0.01 * i, 2) for i in range(20)])
    for t, rec, x, n, up in rows:
        print(f"tau={t:.2f} recall={rec:.3f} false_matches={x}/{n} upper95={up:.4f}")
    print("chosen threshold:", chosen)


def calib_selftest():
    w = World(seed=0)
    rng = np.random.default_rng(2)
    a = rng.integers(0, w.n_intents, 3000)
    sa = w.paraphrase(a)
    same_sims = (sa * w.paraphrase(a)).sum(1)
    b = (a // w.M) * w.M + (a % w.M + rng.integers(1, w.M, 3000)) % w.M
    sib_sims = (sa * w.paraphrase(b)).sum(1)
    sims = np.concatenate([same_sims, sib_sims])
    lab = np.concatenate([np.ones(3000, bool), np.zeros(3000, bool)])
    chosen, rows = calibrate(sims, lab, [0.90, 0.92, 0.93, 0.94, 0.95, 0.96])
    ok = chosen is not None and all(r[4] > 0.01 for r in rows if r[0] < chosen) and rows[[r[0] for r in rows].index(chosen)][4] <= 0.01
    print(f"  [{'PASS' if ok else 'FAIL'}] calibrate() picks tau={chosen}: lowest threshold with Wilson upper bound <= 1% on 3000 sibling pairs")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        calibrate_cli(sys.argv[2], sys.argv[3]); sys.exit(0)
    t0 = time.time()
    print("self-tests:")
    ok = selftests()
    ok = calib_selftest() and ok
    if "--selftest" not in sys.argv:
        experiments()
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}   ({time.time() - t0:.0f} s)")
    sys.exit(0 if ok else 1)
