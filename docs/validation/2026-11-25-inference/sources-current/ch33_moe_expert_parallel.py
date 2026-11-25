"""Chapter 33: MoE expert parallelism -- routing, load balance, all-to-all volume, and a gloo all-to-all check.

Run:  python3 ch33_moe_expert_parallel.py          (the chapter's numbers)
      python3 ch33_moe_expert_parallel.py --test   (self-tests)
CPU only. The all-to-all uses torch.distributed with the gloo backend to verify dispatch/combine LOGIC.
It says nothing about NCCL/NVLink/InfiniBand performance.
"""
import os
import random
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ------------------------------------------------------------- 1. routing and load imbalance
def skewed_router(n_tokens, n_experts, topk, skew, seed):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n_experts, generator=g) * skew                  # some experts are globally "popular"
    logits = torch.randn(n_tokens, n_experts, generator=g) + base
    return logits.topk(topk, dim=-1).indices                           # [n_tokens, topk]


def expert_loads(assign, n_experts):
    return torch.bincount(assign.flatten(), minlength=n_experts).float()


def gpu_loads_contiguous(loads, n_gpus):
    return loads.reshape(n_gpus, -1).sum(1)


def eplb_place(loads, n_gpus, n_redundant):
    """Greedy placement with replicas: replicate the hottest experts, then pack replicas onto the lightest GPU.
    Each replica of an expert takes an equal share of that expert's load."""
    E = len(loads)
    if n_gpus <= 0 or n_redundant < 0 or E == 0 or (E+n_redundant) % n_gpus:
        raise ValueError("equal-slot placement requires positive GPUs and divisible expert copies")
    copies = [1] * E
    for _ in range(n_redundant):                                       # give the next copy to the expert with
        e = max(range(E), key=lambda i: loads[i] / copies[i])          # the largest per-copy load
        copies[e] += 1
    items = sorted(((loads[e] / copies[e], e) for e in range(E) for _ in range(copies[e])), reverse=True)
    gpu_load = [0.0] * n_gpus
    per_gpu_count = [0] * n_gpus
    slots = (E + n_redundant) // n_gpus
    for w, _e in items:
        cands = [g for g in range(n_gpus) if per_gpu_count[g] < slots]
        g = min(cands, key=lambda i: gpu_load[i])
        gpu_load[g] += w
        per_gpu_count[g] += 1
    return torch.tensor(gpu_load)


def imbalance(x):
    return float(x.max() / x.mean())


# ------------------------------------------------------------- 2. all-to-all volume
def alltoall_bytes_per_gpu(tokens_per_gpu, topk, hidden, bytes_per_elem, ep):
    """Dispatch bytes sent by one GPU (combine is the same again). Fraction (ep-1)/ep of routed copies leave the GPU
    if routing is uniform over GPUs."""
    return tokens_per_gpu * topk * hidden * bytes_per_elem * (ep - 1) / ep


# ------------------------------------------------------------- 3. all-to-all dispatch/combine on gloo
def _worker(rank, world, init_file, n_tokens, hidden, n_experts, topk, result_q, mode="uniform"):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    g = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(n_tokens, hidden, generator=g)
    per_rank_ = n_experts // world
    # mode "uniform": any expert; "all_to_rank0": every copy goes to rank 0 (other ranks receive nothing)
    hi = n_experts if mode == "uniform" else per_rank_
    experts = torch.randint(0, hi, (n_tokens, topk), generator=g)
    weights = torch.softmax(torch.randn(n_tokens, topk, generator=g), -1)
    per_rank = n_experts // world

    def expert_fn(e, t):                                               # a trivial "expert": scale by (e+1)
        return t * (e + 1)

    # flatten (token, slot) copies and sort by destination rank
    tok_idx = torch.arange(n_tokens).repeat_interleave(topk)
    exp_flat = experts.flatten()
    dest = exp_flat // per_rank
    order = torch.argsort(dest, stable=True)
    send_x = x[tok_idx][order]
    send_e = exp_flat[order]
    send_counts = torch.bincount(dest, minlength=world)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)                   # exchange the sizes first
    s, r = send_counts.tolist(), recv_counts.tolist()
    recv_x = torch.empty(sum(r), hidden)
    recv_e = torch.empty(sum(r), dtype=torch.long)
    dist.all_to_all_single(recv_x, send_x, r, s)                       # dispatch tokens
    dist.all_to_all_single(recv_e, send_e, r, s)                       # and their expert ids
    out = torch.stack([expert_fn(int(e), row) for e, row in zip(recv_e, recv_x)]) if sum(r) else recv_x
    back = torch.empty(sum(s), hidden)
    dist.all_to_all_single(back, out, s, r)                            # combine: return results to owners
    result = torch.empty_like(back)
    result[order] = back                                               # undo the sort
    y = torch.zeros(n_tokens, hidden)
    y.index_add_(0, tok_idx, result * weights.flatten()[:, None])
    # reference on this rank alone
    ref = torch.zeros(n_tokens, hidden)
    for t in range(n_tokens):
        for k in range(topk):
            ref[t] += weights[t, k] * expert_fn(int(experts[t, k]), x[t])
    # ownership: every expert id this rank received must be one this rank actually holds. A stateless toy expert
    # cannot detect mis-routing by itself, so check it explicitly.
    owned = bool(((recv_e // per_rank) == rank).all()) if sum(r) else True
    result_q.put((rank, bool(torch.allclose(y, ref, atol=1e-5)) and owned, s))
    dist.destroy_process_group()


def check_alltoall(world=4, mode="uniform", topk=2, n_tokens=64, n_experts=16):
    with tempfile.TemporaryDirectory() as d:
        init = os.path.join(d, "rdv")
        q = mp.get_context("spawn").Queue()
        procs = [mp.get_context("spawn").Process(target=_worker, args=(r, world, init, n_tokens, 8, n_experts, topk, q, mode))
                 for r in range(world)]
        [p.start() for p in procs]
        res = sorted(q.get() for _ in procs)
        [p.join() for p in procs]
    return res


# ------------------------------------------------------------- self-tests
def run_tests():
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    E, G, T, K = 64, 8, 4000, 2
    a = skewed_router(T, E, K, 0.5, seed=1)
    check("router: shape [T, k], ids in range", tuple(a.shape) == (T, K) and int(a.min()) >= 0 and int(a.max()) < E)
    check("router: the k experts of one token are distinct", bool((a[:, 0] != a[:, 1]).all()))
    check("router: same seed gives the same assignment", bool((a == skewed_router(T, E, K, 0.5, seed=1)).all()))
    loads = expert_loads(a, E)
    check("expert loads sum to tokens x k", int(loads.sum()) == T * K)
    check("contiguous GPU loads conserve the total", float(gpu_loads_contiguous(loads, G).sum()) == float(loads.sum()))
    check("imbalance of a perfectly even load is exactly 1.0", imbalance(torch.ones(8)) == 1.0)
    check("imbalance is never below 1", all(imbalance(gpu_loads_contiguous(expert_loads(skewed_router(T, E, K, sk, 0), E), G)) >= 1.0
                                            for sk in (0.0, 0.5, 1.0)))
    imb = [imbalance(gpu_loads_contiguous(expert_loads(skewed_router(T, E, K, sk, 0), E), G)) for sk in (0.0, 0.5, 1.0)]
    check("more skew, more contiguous imbalance", imb[0] < imb[1] < imb[2])
    hot = expert_loads(skewed_router(T, E, K, 1.0, 0), E)
    p8, p0 = eplb_place(hot, G, 8), eplb_place(hot, G, 0)
    check("placement conserves total load (8 replicas and 0 replicas)",
          abs(float(p8.sum()) - float(hot.sum())) < 1e-3 and abs(float(p0.sum()) - float(hot.sum())) < 1e-3)
    check("replicas improve a skewed load", imbalance(p8) < imbalance(gpu_loads_contiguous(hot, G)))
    check("skew 1.0: +8 and +16 replicas bring imbalance under 1.02", imbalance(p8) < 1.02 and imbalance(eplb_place(hot, G, 16)) < 1.02)
    hc0 = eplb_place(torch.tensor([100., 1., 1., 1.]), 2, 0)
    hc2 = eplb_place(torch.tensor([100., 1., 1., 1.]), 2, 2)
    check("hand case, no replicas: loads [100,1,1,1] on 2 GPUs -> [101, 2]", torch.allclose(hc0, torch.tensor([101., 2.])))
    check("hand case, 2 replicas: hot expert split 3 ways -> [67.667, 35.333]",
          torch.allclose(hc2, torch.tensor([67.6667, 35.3333]), atol=1e-3))
    try:
        eplb_place(hot, G, 3); raised = False
    except ValueError:
        raised = True
    check("replica count that breaks (E+R) % GPUs == 0 raises ValueError (documented failure)", raised)
    check("dispatch bytes are 0 on one GPU", alltoall_bytes_per_gpu(8192, 2, 4096, 2, 1) == 0)
    check("dispatch bytes linear in tokens", alltoall_bytes_per_gpu(2000, 2, 64, 2, 8) == 2 * alltoall_bytes_per_gpu(1000, 2, 64, 2, 8))
    check("EP=8 dispatch is 117.4 MB (chapter figure)", abs(alltoall_bytes_per_gpu(8192, 2, 4096, 2, 8) / 1e6 - 117.44) < 0.01)
    for label, kw in (("4 ranks, uniform routing, top-2", {}),
                      ("2 ranks, top-1", dict(world=2, topk=1)),
                      ("4 ranks, every copy to rank 0 (3 ranks receive nothing)", dict(mode="all_to_rank0"))):
        res = check_alltoall(**kw)
        world = kw.get("world", 4)
        topk = kw.get("topk", 2)
        check(f"gloo all-to-all matches reference: {label}", all(ok for _, ok, _ in res))
        check(f"  send counts conserve tokens x k per rank: {label}", all(sum(sc) == 64 * topk for _, _, sc in res))
        if kw.get("mode") == "all_to_rank0":
            check("  all sends target rank 0", all(sc[0] == 64 * topk and sum(sc[1:]) == 0 for _, _, sc in res))
    for name, good in results:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
    print(f"{sum(g for _, g in results)}/{len(results)} self-tests pass")
    return all(g for _, g in results)


def main():
    E, G, T, K = 64, 8, 20000, 2
    print("== load imbalance (max GPU load / mean), 64 experts on 8 GPUs, top-2 routing ==")
    print("skew  contiguous  +8 replicas  +16 replicas")
    for skew in (0.0, 0.5, 1.0):
        loads = expert_loads(skewed_router(T, E, K, skew, seed=0), E)
        print(f"{skew:4.1f}  {imbalance(gpu_loads_contiguous(loads, G)):10.3f}  "
              f"{imbalance(eplb_place(loads, G, 8)):11.3f}  {imbalance(eplb_place(loads, G, 16)):12.3f}")
    print("\n== all-to-all dispatch volume per GPU per layer (hidden 4096, BF16, top-2, 8192 tokens/GPU) ==")
    for ep, link, bw in ((8, "NVLink 450 GB/s one-way (datasheet)", 450e9), (16, "InfiniBand 50 GB/s (400 Gb/s)", 50e9)):
        b = alltoall_bytes_per_gpu(8192, 2, 4096, 2, ep)
        print(f"EP={ep:2d}: {b/1e6:7.1f} MB dispatch (+ same combine) -> floor {2*b/bw*1e6:8.1f} us over {link}")
    print("\n== gloo all-to-all dispatch/combine vs single-process reference (4 ranks) ==")
    for rank, ok, sends in check_alltoall():
        print(f"rank {rank}: matches reference? {ok}   tokens sent to ranks 0..3 = {sends}")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    main()
