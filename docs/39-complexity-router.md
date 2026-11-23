# Chapter 39 — Multi-Model Complexity Router: Entropy, Calibration, Cost

!!! warning "Evidence status"
    **Run on CPU, in a synthetic world:** the entropy feature, the routing and cost accounting, the calibration procedure, and 17 self-tests (`ch39_complexity_router.py`).
    **Invented:** both "models" and all queries. Each query has a latent difficulty *d*; the small and large models answer correctly with made-up probabilities that depend on *d*; the small model's token logits are synthetic. Every number below is a property of that world, **not of any real model pair**. Section 39.9 is the procedure for measuring the real numbers; it is **UNTESTED** on a real model pair.
    Sample outputs vary with seeds; the self-tests assert properties, not these exact numbers.

A complexity router sends easy queries to a small, cheap model and hard ones to a large, expensive one. The project brief says to route on "entropy". This chapter fixes what that means, shows how to choose the threshold without fooling yourself, and measures when the router actually saves money. In this synthetic world it saves **less than you would hope**, and the reasons are the useful part.

## 39.1 What "entropy" means here

"Entropy" is ambiguous until you fix five things. We fix them:

| Choice | Definition used |
|---|---|
| Which distribution | the **small** model's next-token distribution, `softmax(logits)`, temperature 1 |
| Which positions | the first **K = 8** greedily decoded tokens of the small model's answer |
| Aggregation | **mean** over those K positions |
| Units | **nats** (natural log). Divide by ln 2 for bits |
| Range | 0 ≤ H ≤ ln V; for V = 64, ln V = 4.159 nats |

Formally, for one position with probabilities p_v: H = −Σ_v p_v ln p_v. The router feature is the mean of H over the K positions. A threshold τ (in nats) sends a query to the large model when the feature exceeds τ.

**Consequences of that definition.**

- The small model must run first, for K tokens, to produce the feature. **The small model's cost is always paid.** The large model's cost is paid only for routed queries. Cost relative to always-large is `(c_s + f·c_l)/c_l` where *f* is the routed fraction.
- τ is specific to the small model, its tokenizer and vocabulary, the prompt template, and the sampling temperature. Change any of them and τ must be recalibrated.
- An API that returns only the top-k logprobs gives you a *truncated* entropy (renormalised over k tokens), which is not the same quantity. Use the same truncation at calibration and at serving.

**Hand check.** A three-way distribution (0.5, 0.25, 0.25) has H = 0.5 ln 2 + 2 × 0.25 ln 4 = 0.3466 + 0.6931 = 1.0397 nats, which is 1.5 bits (Exercise 1).

## 39.2 The synthetic world

Difficulty d ~ Beta(2, 3). The small model answers correctly with probability `0.97 − 0.85 d`, the large model with `0.98 − 0.25 d` (both clipped to [0.02, 1]). The small model's logits are Gaussian noise plus a boost on one token; the boost shrinks with *d* and carries per-query noise, so easy queries tend to give peaked, low-entropy distributions and the feature is informative but imperfect. A second, weaker baseline feature, prompt length, is also generated.

## 39.3 Self-tests (17, all pass)

| Group | What is checked |
|---|---|
| Entropy (6) | uniform = ln V; one-hot = 0; 0 ≤ H ≤ ln V on random logits; shift invariance; fair coin = ln 2; flatter means larger H |
| World (2) | corr(H, d) is positive and below 1; large model beats small |
| Routing (3) | routed fraction nonincreasing in τ; τ = −∞ equals always-large exactly; τ = +∞ equals always-small exactly |
| AUC (2) | oracle ≥ entropy > length > 0.5; a constant score has AUC exactly 0.5 |
| Cost (2) | hand computation; routing everything costs more than always-large |
| Calibration (2) | the chosen τ meets its own validation criterion; conservative τ routes at least as many queries as naive |

## 39.4 How informative is the feature?

Sample output on 60 000 queries:

| quantity | value |
|---|---|
| always-small quality | 0.628 |
| always-large quality | 0.878 |
| corr(H, d) | 0.784 |
| AUC of oracle *d* for "small model wrong" | 0.706 |
| AUC of entropy | 0.668 |
| AUC of prompt length | 0.630 |

AUC 0.668 means: pick one query the small model got wrong and one it got right; the wrong one has the higher entropy 66.8% of the time. Even the oracle that knows *d* exactly reaches only 0.706, because in this world the small model errs at random even on easy queries (its accuracy at *d* = 0 is 0.97, at *d* = 0.5 it is 0.545). **A router cannot predict what is not predictable.** This ceiling decides everything below.

## 39.5 The cost–quality frontier

Quality as a function of the routed fraction, c_small = 1, c_large = 15 (cost ratio is relative to always-large):

| routed | oracle *d* | entropy | length | random | cost ratio |
|---|---|---|---|---|---|
| 0.10 | 0.675 | 0.670 | 0.667 | 0.652 | 0.167 |
| 0.20 | 0.714 | 0.707 | 0.701 | 0.677 | 0.267 |
| 0.30 | 0.747 | 0.741 | 0.731 | 0.703 | 0.367 |
| 0.40 | 0.777 | 0.769 | 0.758 | 0.727 | 0.467 |
| 0.50 | 0.803 | 0.794 | 0.785 | 0.752 | 0.567 |
| 0.70 | 0.845 | 0.837 | 0.828 | 0.802 | 0.767 |
| 1.00 | 0.878 | 0.878 | 0.878 | 0.878 | 1.067 |

Entropy beats random and beats prompt length, and sits close to the oracle. But all curves are only modestly above the straight line from (0, 0.628) to (1, 0.878). At 50% routed, entropy gets 0.794 against random's 0.752: 4.2 points of quality for the same cost. Note the last row: routing everything costs **1.067**, more than always-large, because the small model ran first.

## 39.6 Calibrating the threshold without fooling yourself

The goal: the cheapest router whose quality is within δ = 0.02 of always-large. Procedure:

1. Split labelled data into **validation** (choose τ) and **test** (report). Never report on the data that chose τ.
2. For each candidate τ, compute the per-query **paired gap** `router_correct − large_correct`. Pairing matters: both models answered the same queries, so the gap's standard error is much smaller than comparing two independent accuracies.
3. **Naive:** take the largest τ with mean gap ≥ −δ.
   **Conservative:** take the largest τ with `mean gap − 1.645·SE ≥ −δ` (one-sided 95% lower bound).

Result over 300 random 50/50 splits of 4000 queries (2000 validation, 2000 test):

| method | mean test gap | P(test gap < −0.02) | routed | cost ratio |
|---|---|---|---|---|
| naive | −0.0207 | **0.547** | 0.826 | 0.893 |
| conservative | −0.0143 | 0.090 | 0.872 | 0.938 |

The naive threshold sits exactly on the boundary, so on fresh data it violates the target **54.7% of the time**: scanning many thresholds and keeping the best one on validation is a winner's curse. The conservative rule cuts violations to 9.0%. It does not reach 5% because the lower bound is applied at one threshold per validation set but chosen out of a scan of 200, and because 2000 validation queries is small. More validation data, or a margin on top, lowers it further. It costs about 4.5 points of cost ratio (0.893 → 0.938) to buy that safety. That price is real and should be a stated decision.

## 39.7 Distribution shift

Threshold calibrated on Beta(2,3) difficulty (τ = 1.408 nats), deployed on other traffic:

| traffic | routed | quality | always-large | gap |
|---|---|---|---|---|
| same, Beta(2,3) | 0.841 | 0.859 | 0.877 | −0.0185 |
| harder, Beta(3,2) | 0.964 | 0.821 | 0.827 | −0.0060 |
| much harder, Beta(5,2) | 0.993 | 0.800 | 0.801 | −0.0016 |
| easier, Beta(2,5) | 0.724 | 0.881 | 0.907 | **−0.0266** |

Harder traffic is safe but expensive: the router escalates almost everything, so the saving vanishes. **Easier** traffic breaches the target (−0.0266 against −0.02): the easy queries the small model gets wrong are exactly the ones the feature cannot flag. So the failure mode to monitor is a drift toward easier traffic *and* the quality of the unrouted slice. Re-run calibration on a schedule and on every change of small model, prompt template or temperature.

## 39.8 When does the router pay at all?

At the calibrated routed fraction f = 0.84, the router costs `(1 + 0.84 c_l/c_s)/(c_l/c_s)` of always-large, which is below 1 only if c_l/c_s > 1/(1 − f) = 6.25 (the printed 6.24 uses the unrounded f).

| c_l / c_s | cost ratio | verdict |
|---|---|---|
| 2 | 1.340 | costs more than always-large |
| 5 | 1.040 | costs more |
| 10 | 0.940 | saves 6% |
| 15 | 0.906 | saves 9% |
| 30 | 0.873 | saves 13% |
| 100 | 0.850 | saves 15% |

**Hand check, c_l/c_s = 15:** (1 + 0.84 × 15)/15 = 13.6/15 = 0.9067 (table 0.906, rounding of f).

Honest conclusion for this world: holding quality within 2 points of the large model, a feature this weak routes 84% of traffic upward and saves under 10% at a 15× price gap. A complexity router earns its keep only when (a) the small model's errors are predictable (AUC well above 0.67), (b) a large fraction of traffic is genuinely easy, and (c) the price gap is large. Measure all three before building it.

## 39.9 Runbook: measuring it on a real model pair (UNTESTED)

1. **Sample traffic.** At least 3000 representative queries, stratified by task type. Run **both** models on all of them.
2. **Judge.** Score each answer with a rubric, exact match, or a held-out judge model; spot-check by hand. The router can only be as good as this label.
3. **Log the feature.** For the small model, request logprobs for the first K tokens. With vLLM, use `logprobs` on the OpenAI-compatible endpoint; note that servers return only top-k logprobs, so compute the entropy of the renormalised top-k and use the **same** k everywhere. Record per query: feature, small correct, large correct, prompt length, task type.
4. **Check the ceiling first.** Compute the AUC of the feature for "small model wrong". If it is near 0.5, stop: no threshold will help.
5. **Calibrate** with the conservative rule of 39.6 on a validation half and report on the test half. Use at least 2000 validation queries; more if the target δ is tight.
6. **Price it.** Plug real per-request costs into `cost_ratio`, including the small model's always-paid cost, and check the break-even in 39.8.
7. **Shadow, then canary.** Log what the router would have done while serving everything from the large model. Then canary 1–5% with a kill switch, and track unrouted-slice quality separately from the overall figure.

## 39.10 Failure catalogue

| Failure | Cause | Defence |
|---|---|---|
| Threshold violates target in production | naive calibration; winner's curse | conservative rule; separate test set |
| Silent quality loss on easy traffic | confident small model that is wrong | audit the unrouted slice; add a cheap verifier for high-stakes classes |
| Saving disappears on hard traffic | router escalates nearly everything | track routed fraction; alert when it passes ~90% |
| τ stops working after a change | entropy depends on model, prompt template, temperature, tokenizer | recalibrate on any change |
| Router costs more than always-large | small model cost always paid; low price gap | compute break-even before building |
| Truncated logprobs | API returns top-k only | calibrate and serve with the same k |
| Judge error | label noise treated as truth | check judge against human labels on a sample |

## 39.11 Exercises

1. Compute the entropy of (0.5, 0.25, 0.25) in nats and in bits.
2. The calibrated threshold is 1.408 nats. Express it in bits, and say what a feature of 1.408 nats means for a one-hot-plus-noise distribution over 64 tokens.
3. What does the router cost relative to always-large if it routes every query, with c_small = 1, c_large = 15?
4. Why does the naive threshold violate the target 54.7% of the time rather than about 5%?
5. Easier traffic broke the target but harder traffic did not. Explain, and say what you would monitor.
6. Entropy AUC is 0.668. If you could raise it to 0.85 with a better feature, what two quantities in the cost table would change?
7. **Code.** Add a cascade variant: the small model answers; a verifier returns a confidence; low-confidence answers go to the large model. Write the cost formula, including the verifier.

### Solutions

1. H = 0.5 ln 2 + 0.25 ln 4 + 0.25 ln 4 = 0.3466 + 0.3466 + 0.3466 = 1.0397 nats. Divide by ln 2 = 0.6931: 1.5 bits. (Check: 0.5 × 1 + 0.5 × 2 = 1.5 bits.)
2. 1.408 / 0.6931 = 2.031 bits. Compared with the maximum ln 64 = 4.159 nats (6 bits), the threshold sits at about a third of the maximum. It corresponds to a distribution that is clearly peaked but not confident: roughly the mass spread over four to five tokens (2^2.03 ≈ 4.1 equally likely tokens, the "perplexity" of the position).
3. (1 + 1 × 15)/15 = 1.0667, i.e. 6.7% more than always-large, because the small model ran first for nothing.
4. The threshold is chosen as the boundary of "mean gap ≥ −δ" among 200 candidates, so by construction its validation gap sits at the edge, with noise pushing it favourably (selection on noise). On fresh data the noise is independent, so the true gap is below the boundary about half the time or more. Requiring a lower confidence bound removes most, not all, of that.
5. Easy queries the small model gets wrong have low entropy, so the feature does not flag them, and in easier traffic they are a larger share of the unrouted slice. Harder traffic raises the routed fraction (more queries exceed τ), which protects quality but removes the saving. Monitor routed fraction, quality of the unrouted slice (sampled and judged), and the feature distribution against the calibration set.
6. The routed fraction needed to reach the quality target would fall (fewer queries need escalating), which lowers the cost ratio and moves the break-even price ratio 1/(1 − f) down toward 1. The always-large and always-small quality columns do not change.
7. Cost per request = c_small + c_verifier + f_escalated × c_large, relative to always-large: (c_s + c_v + f·c_l)/c_l. It pays only if the verifier is cheap relative to the large model and its errors are less correlated with the small model's than entropy is. Measure verifier AUC the same way as in 39.4.

## 39.12 Summary

- Fix the definition: which model, which tokens, which aggregation, which units. τ is meaningless without them.
- The small model's cost is always paid; routing everything costs more than not routing.
- Calibrate on held-out data with a paired, conservative bound. The naive choice failed its target 55% of the time here.
- The ceiling is how predictable the small model's errors are. In this world it was low, and the router saved under 10%.
- All numbers here are synthetic. Run Section 39.9 on your real model pair before building anything on this.

## Files

- Code: `docs/examples/inference-systems/ch39_complexity_router.py`
