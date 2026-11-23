"""Chapter 33: MoE expert parallelism -- routing, load balance, all-to-all volume, and a gloo all-to-all check.

Run:  python3 ch33_moe_expert_parallel.py
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
def _worker(rank, world, init_file, n_tokens, hidden, n_experts, topk, result_q):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    g = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(n_tokens, hidden, generator=g)
    experts = torch.randint(0, n_experts, (n_tokens, topk), generator=g)
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
    result_q.put((rank, bool(torch.allclose(y, ref, atol=1e-5)), s))
    dist.destroy_process_group()


def check_alltoall(world=4):
    with tempfile.TemporaryDirectory() as d:
        init = os.path.join(d, "rdv")
        q = mp.get_context("spawn").Queue()
        procs = [mp.get_context("spawn").Process(target=_worker, args=(r, world, init, 64, 8, 16, 2, q))
                 for r in range(world)]
        [p.start() for p in procs]
        res = sorted(q.get() for _ in procs)
        [p.join() for p in procs]
    return res


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
    main()
