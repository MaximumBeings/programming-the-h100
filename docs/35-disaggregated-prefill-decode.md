# Chapter 35: Disaggregated Prefill/Decode over RDMA

!!! abstract "Goal and prerequisites"

    Learn why serving systems sometimes split prefill and decode onto separate GPU pools, what it costs to move the KV cache between them, and how to decide whether the split pays off. Read [Chapter 11](11-multi-gpu-nccl.md) (NVLink, InfiniBand), [Chapter 32](32-inference-engine-and-harness.md) (the harness) and [Chapter 34](34-kv-cache-offloading.md) (KV size and transfer reasoning) first.

!!! warning "Evidence level"

    CPU arithmetic and a **simulation** that reuses Chapter 32's placeholder cost model. Hardware bandwidths and capacities are datasheet values; check yours. The simulation shows a mechanism, not the performance of a real cluster, and its pool sizes are not tuned. No RDMA transfer was performed.

## 35.1 Why split the phases

Prefill is compute-bound and decode is memory-bound ([Chapter 32](32-inference-engine-and-harness.md)). On one GPU they interfere: when a new prompt's prefill runs inside a batch, every decoding user waits for it, so inter-token latency spikes. Disaggregation gives each phase its own pool, so prefill bursts do not stall decode, and each pool can be sized and configured for its own bottleneck.

The price is that every request's KV cache, built on a prefill GPU, must reach a decode GPU.

```
 request -> [prefill pool] --- KV cache over RDMA/NVLink ---> [decode pool] -> tokens
             compute-bound                                     memory-bound
             sets TTFT                                         sets TPOT
```

## 35.2 What the transfer costs

The size is the KV arithmetic from earlier chapters: 320 KiB per token for a Llama-3-70B-shaped model. Moving it over a network link takes `bytes / bandwidth`.

### Worked Example 35.2.1

```
== KV transfer time for one prompt (no overlap) and EXPOSED time with ideal layer-wise streaming ==
tokens   KV size     100 Gb/s   200 Gb/s   400 Gb/s    exposed@400 (prefill 20k tok/s)
  1024     0.34 GB      26.8       13.4        6.7 ms       0.0 ms
  4096     1.34 GB     107.4       53.7       26.8 ms       0.0 ms
 16384     5.37 GB     429.5      214.7      107.4 ms       0.0 ms
 65536    21.47 GB    1718.0      859.0      429.5 ms       0.0 ms
```

A 4,096-token prompt costs 27 ms on a 400 Gb/s link and 107 ms on a 100 Gb/s link. That is added to the first inter-token gap unless it is hidden.

The "exposed" column assumes **ideal layer-wise streaming**: each layer's KV is sent as soon as prefill finishes that layer, so the transfer overlaps prefill and only the excess is visible. With prefill at an assumed 20,000 tokens per second the whole transfer fits behind it, so the exposed time is zero. That is a best case. Real transfers contend with other traffic, need both ends ready, and may use different tensor-parallel layouts that require re-sharding the KV before it can be used. Measure it.

Practical transports for this are GPUDirect RDMA over InfiniBand or RoCE, and NVLink inside a node. Systems that implement the idea include DistServe, Splitwise, Mooncake and NVIDIA's Dynamo with its NIXL transfer library; read their papers and documentation for the details this chapter omits.

## 35.3 The hardware mapping is a hypothesis

The LinkedIn project description pairs H100s for prefill with L40S GPUs for decode. Test that with arithmetic before building it. Decode is bandwidth-bound, and the two cards differ sharply:

### Worked Example 35.3.1

70B weights in FP8 (70 GB), 4,096-token sequences, datasheet bandwidth and capacity:

```
config          decode floor    KV room   max 4K-token seqs (upper bound)
2 x H100 SXM       10.4 ms/tok    90.0 GB     67
2 x L40S           40.5 ms/tok    26.0 GB     19
4 x L40S           20.3 ms/tok   122.0 GB     90
```

The L40S has about 864 GB/s of memory bandwidth against about 3.35 TB/s on an H100 SXM, so on the same number of GPUs its decode floor is four times worse, and it has 48 GB instead of 80 GB. It also lacks NVLink, so tensor parallelism runs over PCIe. Four L40S have a decode floor about twice that of two H100s (20.3 against 10.4 ms) while holding more sequences, so the case for them is **cost**, not speed: it holds only if the L40S cost enough less per unit of served throughput, after the all-reduce overhead of tensor parallelism over PCIe.

Treat "H100 for prefill, cheaper GPU for decode" as a cost hypothesis to measure. The capacity column is an upper bound that ignores activations and framework overhead.

Prefill and decode may also want different parallelism layouts (for example a different tensor-parallel degree), which means the KV cache must be re-sharded on arrival.

## 35.4 When does splitting win? A simulation

The simulation compares, on the **same total number of GPUs**, two colocated instances (requests alternate between them, prefill runs inside the decode iteration) against one prefill instance plus one decode instance. The decode instance never pays prefill time; the prefill instance emits the first token; the KV transfer uses 400 Gb/s. Prompts have median 2,048 tokens and outputs median 128, with Poisson arrivals.

```
rate  mode          ttft_p99   tpot_p99   slo_ok(ttft<=2s,tpot<=0.1s)
   6  colocated x2     0.379     0.0364      1.00
   6  disagg 1+1       0.474     0.0287      1.00
  10  colocated x2     0.477     0.0508      1.00
  10  disagg 1+1       0.516     0.0341      1.00
  14  colocated x2     1.272     0.0736      1.00
  14  disagg 1+1       1.042     0.1351      0.96
```

Read it as three lessons:

1. **Decode is protected.** At 6 and 10 requests per second the disaggregated TPOT p99 is 21 to 33 percent lower (0.0287 against 0.0364, and 0.0341 against 0.0508), because decode iterations no longer absorb prefill work. This is the benefit.
2. **It is not free.** At low load TTFT p99 is higher for the split (0.474 against 0.379): all prefill now queues on one instance instead of two, and the transfer adds delay.
3. **The pool ratio matters.** At 14 requests per second the single decode instance becomes the bottleneck and TPOT p99 is much worse for the split (0.1351 against 0.0736), and SLO attainment falls to 0.96. A 1:1 split is not the right ratio for this workload; the prefill-to-decode ratio has to follow the workload's prompt and output lengths.

These numbers come from placeholder constants. The mechanism (interference removed, a transfer and a queue added, a ratio to tune) is the content; the values are not.

## 35.5 Decision checklist

| Question | If yes | If no |
|---|---|---|
| Are prompts long relative to outputs? | Prefill interference is large; splitting may help | Colocated is simpler and likely as good |
| Is a tight TPOT service-level objective violated by prefill bursts? | Split protects it | Keep colocated, consider chunked prefill |
| Is there a fast link (NVLink, or 200 to 400 Gb/s RDMA)? | Transfer can be hidden | Transfer becomes TTFT |
| Can both pools be kept busy? | Utilization improves | Idle pools waste GPUs |
| Do prefill and decode need different layouts? | Budget for re-sharding | Simpler |

Also compare against cheaper alternatives first: chunked prefill (splitting long prompts so decode iterations interleave) removes much of the interference without a transfer.

## 35.6 What to measure on real hardware (untested)

!!! warning "UNTESTED ON H100"

    | Measurement | How |
    |---|---|
    | Actual transfer time and exposed time | Time KV movement per prompt length on your link; compare with the table above |
    | Interference | TPOT p99 colocated versus split at equal GPU count, using the [Chapter 32](32-inference-engine-and-harness.md) harness |
    | Best pool ratio | Sweep prefill:decode GPU counts at several prompt and output length mixes |
    | Cost | Dollars per million tokens at a fixed SLO, including any cheaper decode GPU |
    | Re-sharding | Time to convert KV between different parallel layouts, if they differ |
    | Failure behavior | What happens when a decode node is lost mid-transfer |

## Chapter Summary

Disaggregation puts compute-bound prefill and memory-bound decode on separate pools so prefill bursts cannot stall decoding users, at the cost of moving each request's KV cache between pools. For a 4,096-token prompt that transfer is about 27 ms on a 400 Gb/s link and 107 ms on 100 Gb/s, and it can be hidden behind prefill only with layer-wise streaming in the best case. The pairing of H100s with L40S GPUs is a cost hypothesis: an L40S has about a quarter of an H100's memory bandwidth and no NVLink, so matching an H100's decode floor takes about four times as many of them. In a simulation on equal GPU counts, the split lowered TPOT p99 by 21 to 33 percent at moderate load, raised TTFT p99 at low load, and lost badly once its single decode instance saturated, so the pool ratio must be tuned. Decide with a measured comparison, and try chunked prefill first.

## Self-Check Questions

1. What is the one thing disaggregation buys and the one thing it costs?
2. Why can the KV transfer in the table be hidden behind prefill, and what does that assume?
3. Why is "H100 for prefill, L40S for decode" a cost hypothesis and not a performance claim?
4. In the simulation, why did the split have worse TTFT at low load?
5. Why did the split fail at 14 requests per second, and what would you change?

## Worked Solutions

1. It removes prefill interference from decode, protecting inter-token latency. It costs a KV-cache transfer for every request, plus pool-ratio and layout complexity.
2. If each layer's KV is sent as soon as that layer finishes prefill, the transfer proceeds during the remaining prefill and only the part beyond prefill's duration is visible. It assumes a fast, uncontended link and compatible layouts at both ends, which real systems may not have.
3. The L40S has about 864 GB/s of bandwidth against 3.35 TB/s and lacks NVLink, so it decodes more slowly per GPU. It can only win if enough of them cost less than the H100s they replace for the same throughput.
4. All prefill work queued on a single prefill instance instead of being spread over two, and each request also waited for its KV transfer before decode could start.
5. The single decode instance, handling twice the work per GPU of the colocated case, became the bottleneck and its batch limit made requests wait. Give decode a larger share of the GPUs and re-run the sweep to find the ratio that suits the workload.
