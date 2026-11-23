# Chapter 33: MoE Expert Parallelism: Sharding, All-to-All and Load Balance

!!! abstract "Goal and prerequisites"

    Learn how a Mixture-of-Experts (MoE) layer is served across several GPUs: which GPU holds which expert, how tokens travel to their experts and back, how much that costs, and why load imbalance, not arithmetic, is usually the problem. Read [Chapter 11](11-multi-gpu-nccl.md) (collectives, NVLink, InfiniBand) and [Chapter 32](32-inference-engine-and-harness.md) (the measurement harness) first.

!!! warning "Evidence level"

    Everything here ran on a **CPU**. The routing study is a simulation with a synthetic skew. The all-to-all check uses PyTorch's **gloo** backend on four CPU processes and verifies that the dispatch and combine **logic** returns the right answer. It does not measure NCCL, NVLink or InfiniBand performance. The bandwidth figures are datasheet values turned into lower bounds, not measurements.

## 33.1 What an MoE layer does

A dense feed-forward layer sends every token through the same weights. An MoE layer holds many expert networks and a small **router**. For each token the router scores all experts and sends the token to its top-`k` (often 2). The experts' outputs are combined, weighted by the router scores.

```
 token -> router -> top-2 experts: E17 (weight 0.7), E41 (weight 0.3)
                       |                       |
                     run E17                 run E41
                       +---------- weighted sum ----------> output
```

Only `k` of the `E` experts run for each token, so an MoE model has many parameters but modest compute per token. The cost is that the experts are too large for one GPU, so they are **sharded**: each GPU holds a subset. This is **expert parallelism**.

## 33.2 The communication pattern

With experts spread across `EP` GPUs, a token may need an expert that lives elsewhere. Each MoE layer therefore does two **all-to-all** exchanges:

1. **Dispatch:** every GPU sends each token copy to the GPU that owns its chosen expert.
2. **Combine:** after the experts run, results travel back to the token's home GPU.

```
 GPU0 tokens ---+--> GPU0 experts            each GPU sends some tokens to every GPU
 GPU1 tokens ---+--> GPU1 experts   dispatch   (all-to-all), computes its experts,
 GPU2 tokens ---+--> GPU2 experts   -------->   then sends results back (combine)
 GPU3 tokens ---+--> GPU3 experts
```

Both exchanges sit on the critical path of every MoE layer, so their cost multiplies by the number of MoE layers in the model.

### Worked Example 33.2.1: dispatch volume and a bandwidth floor

If routing spreads tokens evenly, a fraction `(EP-1)/EP` of each GPU's routed copies leaves the GPU. With 8,192 tokens per GPU, top-2 routing, hidden size 4,096 and BF16 activations:

```
== all-to-all dispatch volume per GPU per layer (hidden 4096, BF16, top-2, 8192 tokens/GPU) ==
EP= 8:   117.4 MB dispatch (+ same combine) -> floor    522.0 us over NVLink 450 GB/s one-way (datasheet)
EP=16:   125.8 MB dispatch (+ same combine) -> floor   5033.2 us over InfiniBand 50 GB/s (400 Gb/s)
```

Two things to read off:

- The data volume barely changes with `EP` (117 versus 126 MB). The **link** it crosses changes everything: about 0.5 ms inside an NVLink-connected node, about 5 ms once the traffic must cross a 400 Gb/s InfiniBand link. That is a tenfold penalty per layer, multiplied by every MoE layer.
- These are floors from datasheet bandwidth. Real all-to-all achieves a fraction of link bandwidth, and the dispatch must also be overlapped with compute to be hidden at all. The NVLink figure is the one-way rate per GPU; check it against your system's documentation.

This is why large MoE serving keeps expert parallelism inside an NVLink domain where possible, and why dispatch kernels are tuned separately for the latency-sensitive decode case and the bandwidth-heavy prefill case.

## 33.3 Load imbalance

The router is not uniform. Some experts are popular, so the GPUs holding them receive more tokens and the others wait. Every layer finishes only when the slowest GPU does, so what matters is the **imbalance**: the busiest GPU's load divided by the average.

### Worked Example 33.3.1: simulated skew

The simulation routes 20,000 tokens to 64 experts on 8 GPUs with top-2 routing. A `skew` parameter adds a per-expert popularity bias to the router scores. `contiguous` puts experts `0-7` on GPU 0, `8-15` on GPU 1 and so on. The other columns replicate the hottest experts and pack the copies onto the lightest GPU.

```
== load imbalance (max GPU load / mean), 64 experts on 8 GPUs, top-2 routing ==
skew  contiguous  +8 replicas  +16 replicas
 0.0       1.021        1.002         1.002
 0.5       2.224        1.004         1.005
 1.0       3.935        1.001         1.000
```

With no skew the contiguous placement is nearly balanced (1.02). With strong skew the busiest GPU carries almost four times the average, so the layer runs at roughly a quarter of the efficiency. Replicating hot experts brings every row close to 1.0.

!!! warning "What this simulation assumes"

    - **Replicas split load perfectly.** Each copy of an expert is assumed to receive an equal share of its tokens. Real routing needs a policy that actually achieves that.
    - **Replicas cost memory.** Each extra copy is another copy of the expert's weights; the run above spends 8 or 16 extra expert slots across the cluster.
    - **The skew is synthetic.** A real model's expert popularity depends on the data and changes over time, so placement must be re-balanced from measured load.

!!! warning "Training versus serving"

    Auxiliary load-balancing losses act on the router during **training**. At inference the router is fixed, so balance has to come from **placement**: expert replication, re-placement from live statistics (often called expert-parallel load balancing, EPLB), and capacity limits. A balanced training loss does not guarantee a balanced serving workload.

## 33.4 Verifying the dispatch and combine logic

An all-to-all is easy to get subtly wrong: send counts must match receive counts, token order must be restored, and results must be weighted and summed back to the right token. The check below runs four processes with the gloo backend. Each rank routes 64 tokens to 16 experts (4 per rank) with top-2 routing, exchanges sizes, dispatches tokens and expert ids, applies a trivial expert (multiply by `expert_id + 1`), returns results, restores the original order and sums with the router weights. Each rank compares the outcome with a single-process reference.

```
== gloo all-to-all dispatch/combine vs single-process reference (4 ranks) ==
rank 0: matches reference? True   tokens sent to ranks 0..3 = [39, 26, 30, 33]
rank 1: matches reference? True   tokens sent to ranks 0..3 = [27, 30, 41, 30]
rank 2: matches reference? True   tokens sent to ranks 0..3 = [39, 27, 35, 27]
rank 3: matches reference? True   tokens sent to ranks 0..3 = [34, 33, 32, 29]
```

The send counts differ by rank and by destination, which is why the exchange needs `all_to_all_single` with **uneven split sizes** and a preliminary exchange of those sizes. A fixed-size all-to-all would silently truncate or pad.

### The pattern in code

```python
dest = exp_flat // per_rank                     # which rank owns each chosen expert
order = torch.argsort(dest, stable=True)        # group token copies by destination
send_counts = torch.bincount(dest, minlength=world)
dist.all_to_all_single(recv_counts, send_counts)            # 1. exchange sizes
dist.all_to_all_single(recv_x, send_x, r, s)                # 2. dispatch tokens
...                                                         # 3. run local experts
dist.all_to_all_single(back, out, s, r)                     # 4. combine: return results
result[order] = back                                        # 5. undo the sort
y.index_add_(0, tok_idx, result * weights.flatten()[:, None])   # 6. weighted sum per token
```

On GPUs the same structure uses NCCL all-to-all or a specialized dispatch library, and the experts run as one **grouped GEMM** so that many small matrix multiplies of different sizes execute in a single kernel launch. Chapters [20](20-tiled-copy.md) to [29](29-workbook-capstone.md) cover the GEMM machinery that a grouped GEMM builds on.

## 33.5 What to measure on real hardware (untested)

!!! warning "UNTESTED ON H100"

    None of the following was run. It is the procedure for turning this chapter into a published result.

| Measurement | How | Why |
|---|---|---|
| Layer time split | Nsight Systems timeline ([Chapter 14](14-nsight-systems.md)): expert compute versus all-to-all | Shows the communication share |
| Imbalance | Per-GPU token counts per layer from the router, max divided by mean | The slowest-GPU penalty |
| Scaling | Tokens per second as `EP` grows within and then across nodes | Finds where InfiniBand hurts |
| Overlap | Does dispatch overlap compute, or run in series? | Hidden versus exposed communication |
| Placement | Imbalance and throughput with and without replicas under real traffic | Whether replication pays for its memory |
| Decode versus prefill | Separate runs at small and large batch | The two regimes need different dispatch strategies |

Report each against the harness from [Chapter 32](32-inference-engine-and-harness.md): fixed workload, TTFT, TPOT and cost, not kernel times alone.

## Chapter Summary

An MoE layer routes each token to its top-`k` experts, and with experts sharded across GPUs every layer does a dispatch and a combine all-to-all on the critical path. The data volume per GPU is nearly independent of the expert-parallel degree, but the link it crosses is not: a bandwidth floor of about 0.5 ms over NVLink becomes about 5 ms over a 400 Gb/s InfiniBand link for the same tensors, per layer. Popular experts overload some GPUs, and the busiest GPU sets the layer time; in a synthetic simulation contiguous placement reached 3.9 times the mean load, and replicating hot experts restored 1.0, at the price of extra weight memory and an assumption of perfectly shared load. Inference balance comes from placement, not from a training loss. A four-process gloo test confirmed that uneven-size dispatch, expert computation, combine and weighted summation reproduce a single-process reference; GPU performance is left for you to measure.

## Self-Check Questions

1. Why does each MoE layer need two all-to-all exchanges?
2. In Worked Example 33.2.1 the dispatch volume barely changes between `EP=8` and `EP=16`. Why does the time floor change by about ten times?
3. Why is the layer time set by the most loaded GPU rather than by the average?
4. Why does a balancing loss used in training not guarantee balanced serving?
5. Why does the all-to-all in Section 33.4 first exchange the send counts?

## Worked Solutions

1. Dispatch sends each token copy to the GPU that owns its chosen expert, and combine returns the expert outputs to the token's home GPU so they can be weighted and summed.
2. The per-GPU data volume depends on tokens, `k`, hidden size and precision, and the `(EP-1)/EP` factor changes little. The time depends on the link: the 8-GPU case stays on NVLink at about 450 GB/s one-way, while the 16-GPU case must cross a 400 Gb/s InfiniBand link at about 50 GB/s, about nine times slower.
3. Every GPU must finish its experts before the combine and the next layer can proceed, so the layer completes when the slowest GPU does; extra idle time on the others is wasted.
4. The balancing loss shapes the router during training. At inference the router is fixed and sees different traffic, so some experts can still be much more popular than others. Placement, replication and re-balancing from live statistics address that.
5. Each rank sends a different number of token copies to each destination, so each receiver must know how much to expect before it can allocate its receive buffer and call the uneven-split exchange.
