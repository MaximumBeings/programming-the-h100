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
2. **It is not free.** At low load TTFT p99 is higher for the split (0.474 against 0.379): all prefill now queues on one instance instead of two. The KV transfer does **not** appear in TTFT in this model, because the first token is produced by the prefill instance; it appears in TPOT instead (section 35.7, experiment B). If your design lets the decode side produce the first token, the transfer moves into TTFT.
3. **The pool ratio matters.** At 14 requests per second the single decode instance becomes the bottleneck and TPOT p99 is much worse for the split (0.1351 against 0.0736), and SLO attainment falls to 0.96. A 1:1 split is not the right ratio for this workload; the prefill-to-decode ratio has to follow the workload's prompt and output lengths. Section 35.7 tests that claim: one prefill and **two** decode instances bring TPOT p99 back to 0.0291 and attainment to 1.00 at the same rate, but that is three GPUs against two, and at equal GPU counts the split does not beat colocation once the load is high.

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

## 35.6 Self-tests (25, all pass)

The simulation was extended in this pass: `sim_disagg_pool(reqs, n_prefill, n_decode, ...)` runs several prefill servers and several decode instances, and `sim_disagg` is now its 1+1 case. Two latent defects in the old function were fixed on the way: it **modified the request objects it was given**, and a request with a one-token reply was sent through an extra decode step. The default workload contains no one-token replies (0 in 20,000), so every number printed earlier in this chapter is unchanged; a regression test confirms `pool(1,1)` equals the old behaviour.

Run `python3 /Users/oluwaseyiawoga/Developer/programming-the-h100/docs/examples/inference-systems/ch35_disaggregation.py --test`.

| Group | What is checked |
|---|---|
| Transfer (5) | KV 327,680 bytes per token; 26.8 ms for 4,096 tokens at 400 Gb/s; linear in tokens and inverse in bandwidth; exposed time is 0 for slow prefill, positive for very fast prefill, and never exceeds the full transfer |
| Hardware mapping (3) | the decode floor halves when GPUs double; the printed rows for 2 x H100 and 4 x L40S; 70 GB of weights on one 48 GB card gives negative KV room and 0 sequences while still printing a floor |
| Colocated (2) | one replica equals the plain simulator; two replicas return every request once |
| Disaggregated (12) | input unchanged; every request returned once; TTFT at least the request's own prefill time; decode cannot finish before the KV arrives; a single prefill server is FIFO; determinism; a faster link never raises median TPOT; a one-token reply finishes at its first token; two exact hand cases for a single request; `pool(1,1)` equals the old function; 2+2 completes the same work |
| Chapter claims (3) | at 6 and 10 req/s the split has lower TPOT p99; at 6 req/s it has higher TTFT p99; at 14 req/s its TPOT p99 is worse and attainment lower |

```
25/25 self-tests pass
```

The three "chapter claims" tests turn the prose of section 35.4 into regression tests: if a later change to the cost model reverses a claim, the test fails instead of the text silently going stale.

**The tests can fail.** Three broken copies: first token stamped at the end of decoding fails three tests; removing the FIFO prefill queue fails two; and dropping the transfer time from the decode start passed all 23 tests of the first version, because a 128-token decode takes far longer than any transfer. The two exact single-request cases were added for that reason.

## 35.7 Four more experiments

Prompt median 2,048, output median 128, 500 requests, seed 3, 400 Gb/s link unless stated. The simulator's prefill speed is its default 40,000 tokens per second, which is why a different figure (20,000) appears in the exposed-time column of section 35.2: that column is a standalone arithmetic illustration (`--extras`).

**A. Pool ratio at a fixed GPU count.**

```
GPUs  rate  layout        ttft_p99  tpot_p99  slo_ok
   3    21  colocated x3     0.921    0.0697    1.00
   3    21  disagg 1+2        8.441    0.0297    0.19
   3    21  disagg 2+1        0.396    0.3699    0.65
   4    28  colocated x4     0.534    0.0739    1.00
   4    28  disagg 1+3       14.199    0.0266    0.16
   4    28  disagg 2+2        0.592    0.0751    1.00
   4    28  disagg 3+1        0.303    0.4893    0.52
```

The ratio is the whole story. A wrong ratio collapses the SLO attainment to between 0.16 and 0.65, because the underloaded pool is idle while the overloaded one queues: too few prefill servers (1+2, 1+3) blow up TTFT to 8 and 14 seconds with a protected, fast TPOT, and too few decode instances (2+1, 3+1) do the reverse. With the right ratio (2+2 at 28 req/s) the split matches colocation and **does not beat it** on these constants. In this simulator the case for splitting is protecting TPOT at moderate load, not a general throughput gain.

The 14 req/s case of section 35.4 was left at two GPUs. Giving that workload a third GPU as a second decode instance (1+2) gives TTFT p99 1.042, TPOT p99 0.0291 and attainment 1.00.

**B. Link speed** (1+1, 10 req/s):

```
link Gb/s  ttft_p99  tpot_p50  tpot_p99
       25     0.516    0.0324    0.0494
      100     0.516    0.0311    0.0362
      200     0.516    0.0309    0.0348
      400     0.516    0.0308    0.0341
  ~infinite     0.516    0.0307    0.0339
```

TTFT does not change at all; TPOT p99 rises from 0.0341 to 0.0494 at 25 Gb/s. The reason is the model's definition: the first token comes from prefill, and the transfer sits between the first and second token, so it is amortised into TPOT. Above 100 Gb/s the link stops mattering for this workload.

**C. Prompt length** (rate 6, two GPUs):

```
prompt_med  mode        ttft_p99  tpot_p99  slo_ok
       256  colocated      0.071    0.0244    1.00
       256  disagg 1+1     0.041    0.0274    1.00
      2048  colocated      0.379    0.0364    1.00
      2048  disagg 1+1     0.474    0.0287    1.00
      8192  colocated      6.235    0.2129    0.14
      8192  disagg 1+1    45.299    0.0292    0.08
```

At short prompts there is little interference to remove and the split is slightly worse on TPOT (0.0274 against 0.0244). At 8,192 the mean prompt is about 10,300 tokens, so one prefill server at 40,000 tokens per second needs `6 x 10,296 / 40,000` = 1.54 seconds of work per second and is saturated: TTFT p99 reaches 45 seconds. The decode side is protected (TPOT p99 0.0292 against 0.2129) but the bottleneck has moved to prefill. Colocated also fails here (SLO attainment 0.14), so both designs are overloaded; the split just fails differently.

**D. Seed spread** (rate 10, TPOT p99): the split has the lower TPOT p99 in **8 of 8 seeds** (colocated 0.0483 to 0.0573, split 0.0336 to 0.0526). The headline claim of the chapter does not depend on the seed. The margin does: seed 0 gives 0.0535 against 0.0526, almost a tie, and seed 1 gives 0.0483 against 0.0336.

## 35.8 Failure modes and edge cases

| Situation | What happens | Where it shows |
|---|---|---|
| Wrong pool ratio | one pool idles while the other queues; SLO attainment falls to 0.16 to 0.65 | experiment A |
| A single prefill server at long prompts | saturates: TTFT p99 45 s at prompt median 8,192 | experiment C |
| Slow link | the transfer is amortised into TPOT; 25 Gb/s raises p99 by 45 percent here | experiment B |
| Where the first token is produced | decides whether the transfer counts toward TTFT or TPOT; this model chooses prefill, a real system may not | section 35.2 |
| Weights larger than the decode GPUs' memory | KV room negative, 0 sequences; the floor is still printed and looks plausible | self-test 8 |
| One-token reply | now finishes at its first token; before this pass it was given an extra decode step | self-test 18 |
| Simulator mutating its input | fixed; a second run on the same list used stale `first_token` values | self-test 11 |
| Decode node lost mid-transfer | not modelled; the request must be re-prefilled or the transfer retried | section 35.9 |
| Different tensor-parallel layouts on the two pools | KV must be re-sharded on arrival; not modelled | section 35.3 |

**Not modelled:** KV re-sharding between layouts; link contention among many simultaneous transfers; decode-instance memory limits (only `max_batch`); load balancing across decode instances beyond round-robin on readiness; prefill batching (the prefill server handles one prompt at a time); chunked prefill, which would be the colocated baseline to beat; failures and retries; prefix reuse; and the cost of the prefill and decode GPUs, which would differ in a mixed fleet.

## 35.9 What to measure on real hardware (untested)

!!! warning "UNTESTED ON H100"

    None of the following was run. Commands are from memory; verify against your installed versions.

    | Measurement | How |
    |---|---|
    | Link bandwidth between nodes | `ib_write_bw` (from the `perftest` package) between two hosts, or `nccl-tests` `sendrecv_perf` / `alltoall_perf`; compare with the datasheet figure |
    | Actual transfer time and exposed time | Time KV movement per prompt length on your link; compare with the table above |
    | Interference | TPOT p99 colocated versus split at equal GPU count, using the [Chapter 32](32-inference-engine-and-harness.md) harness |
    | Best pool ratio | Sweep prefill:decode GPU counts at several prompt and output length mixes, as in experiment A |
    | Cost | Dollars per million tokens at a fixed SLO, including any cheaper decode GPU |
    | Re-sharding | Time to convert KV between different parallel layouts, if they differ |
    | Failure behavior | Kill a decode node mid-transfer and record what the client sees |
    | Engine support | Check your engine's documentation for a prefill/decode disaggregation or KV-transfer mode; names and maturity differ between releases |

    To redo the section 35.7 sweep on your numbers, edit `CostModel` in `ch32_inference_harness.py` with values fitted from the Chapter 32 runbook and rerun `--extras`.

## 35.10 Exercises

1. In experiment B, TTFT p99 is identical for every link speed. Explain, and say where the transfer shows up instead. How would the metrics change if the decode instance produced the first token?
2. Compute the prefill server's utilisation at 14 req/s from the mean prompt length (2,574 tokens) and the 40,000 tokens-per-second prefill rate. What does it say about the 1+1 result at that rate?
3. Predict the utilisation of one prefill server at 21 req/s and use it to explain the `disagg 1+2` row of experiment A.
4. The 1+1 row at 14 req/s has TPOT p99 0.1351 but 1+2 has 0.0291. Is it fair to conclude that the split is better than colocation at 14 req/s?
5. What did the original `sim_disagg` do with a request whose output was one token, and why did that not change any number in this chapter?
6. Using `decode_config`, what are the decode floor, KV room and sequence capacity of three L40S? Where does it fall between the two- and four-GPU rows?

### Solutions

1. The model stamps the first token at the end of prefill, before the transfer starts, so TTFT cannot depend on the link. The transfer lies between the first and second token, and TPOT is computed as `(finish - first) / (output - 1)`, so it is spread over the remaining tokens: 25 Gb/s gives TPOT p99 0.0494 against 0.0341 at 400 Gb/s. If the decode side produced the first token, the transfer would delay it and would show up in TTFT (about 54 ms for an 8,192-token prompt at 400 Gb/s, 859 ms at 25 Gb/s).
2. `14 x 2,574 / 40,000` = 0.90. The single prefill server is 90 percent utilised, close to saturation, so queueing is already significant and TTFT p99 is 1.04 s. The prefill side is almost as stretched as the decode side in that row.
3. `21 x 2,574 / 40,000` = 1.35. One server cannot keep up, so its queue grows during the run and TTFT p99 reaches 8.4 s. The decode side, with two instances, is comfortably underloaded, hence the very low TPOT.
4. No. 1+2 uses three GPUs; the colocated comparison at two GPUs is 1.272 s TTFT p99, 0.0736 TPOT p99 and attainment 1.00 (section 35.4). The fair comparisons are at equal GPU counts, which experiment A gives: at 3 GPUs and 21 req/s colocated x3 meets the SLO fully while both splits miss it, and at 4 GPUs the best split (2+2) ties colocated.
5. It sent the request through the decode loop, which only checks completion after incrementing the token count, so a one-token reply was given a second decode step and its `finish` was later than its `first_token`. The default workload contains no one-token replies (0 in 20,000 requests), so no printed value changed. It now finishes at the first token.
6. Floor `70e9 / (3 x 0.864e12)` = 27.0 ms per token, KV room `3 x 48 - 70` = 74.0 GB, capacity 55 sequences. That sits between 2 x L40S (40.5 ms, 26.0 GB, 19) and 4 x L40S (20.3 ms, 122.0 GB, 90), as expected from the linear scaling.

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
4. All prefill work queued on a single prefill instance instead of being spread over two. The KV transfer is not part of TTFT in this simulation (the first token comes from prefill), so it is not the reason; it delays the start of decoding and shows up in TPOT.
5. The single decode instance, handling twice the work per GPU of the colocated case, became the bottleneck and its batch limit made requests wait. Give decode a larger share of the GPUs and re-run the sweep to find the ratio that suits the workload. Section 35.7 does this: 1 prefill + 2 decode fixes the 14 req/s case, at the cost of a third GPU.

## Files and sources

- Code: `docs/examples/inference-systems/ch35_disaggregation.py` (`--test` for the 25 self-tests, `--extras` for experiments A to D). It imports the harness from `ch32_inference_harness.py`, which must be in the same directory.
- GPU figures are datasheet values (H100 SXM 3.35 TB/s and 80 GB; L40S 864 GB/s and 48 GB). The systems named in section 35.2 (DistServe, Splitwise, Mooncake, NVIDIA Dynamo with NIXL) are pointers for further reading; this chapter makes no claim about their internals.
