#!/usr/bin/env python3
"""CPU binary64 attention/online-softmax oracle, not a CUTLASS/GPU execution.

The tiny d=2 example is mathematical; pinned Hopper FMHA kernels require their
supported larger head dimensions, datatype/stride and launch contracts.
"""
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_close(actual, expected):
    if isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), 'Result shape mismatch')
        for a, e in zip(actual, expected):
            check_close(a, e)
    elif not math.isfinite(actual) or not math.isclose(actual, expected,
                                                     rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f'{actual!r} != {expected!r}')


def score_rows(q, k):
    require(q and k and q[0], 'Nonempty Q/K required')
    d = len(q[0])
    require(all(len(row) == d for row in q + k), 'Q/K head dimension mismatch')
    require(all(math.isfinite(x) for row in q + k for x in row), 'Finite inputs required')
    scale = 1.0 / math.sqrt(d)
    return [[math.fsum(x * y for x, y in zip(qi, kj)) * scale for kj in k]
            for qi in q]


def reference_row(logits, values, allowed):
    """Materialized stable row reference; all-masked rows are explicitly rejected."""
    keys = [j for j, valid in enumerate(allowed) if valid]
    require(keys, 'All-masked row is outside this oracle contract')
    maximum = max(logits[j] for j in keys)
    weights = [math.exp(logits[j] - maximum) if allowed[j] else 0.0
               for j in range(len(logits))]
    denominator = math.fsum(weights)
    probabilities = [p / denominator for p in weights]
    output = [math.fsum(p * v[c] for p, v in zip(probabilities, values))
              for c in range(len(values[0]))]
    return output, maximum + math.log(denominator), probabilities


def online_row(logits, values, blocks, allowed=None):
    """Return output, natural-log LSE and every block's (m,l,z,alpha) state."""
    count = len(logits)
    require(count > 0 and len(values) == count and values[0], 'Nonempty matching K/V required')
    width = len(values[0])
    require(all(len(v) == width for v in values), 'Ragged V rows')
    require(all(math.isfinite(x) for x in logits), 'Finite unmasked score inputs required')
    require(all(math.isfinite(x) for v in values for x in v), 'Finite V required')
    allowed = list(allowed) if allowed is not None else [True] * count
    require(len(allowed) == count, 'Mask width mismatch')
    flat = [j for block in blocks for j in block]
    require(sorted(flat) == list(range(count)), 'Blocks must cover each key exactly once')
    maximum, denominator, numerator = -math.inf, 0.0, [0.0] * width
    trace = []
    for block in blocks:
        valid = [j for j in block if allowed[j]]
        if not valid:
            trace.append(dict(keys=list(block), skipped=True, m=maximum,
                              l=denominator, z=numerator[:], alpha=None))
            continue
        new_maximum = max(maximum, max(logits[j] for j in valid))
        alpha = 0.0 if maximum == -math.inf else math.exp(maximum - new_maximum)
        weights = [(j, math.exp(logits[j] - new_maximum)) for j in valid]
        denominator = alpha * denominator + math.fsum(p for _, p in weights)
        numerator = [alpha * numerator[c] + math.fsum(p * values[j][c] for j, p in weights)
                     for c in range(width)]
        maximum = new_maximum
        trace.append(dict(keys=list(block), skipped=False, m=maximum,
                          l=denominator, z=numerator[:], alpha=alpha))
    require(denominator > 0.0, 'All-masked row is outside this oracle contract')
    return [z / denominator for z in numerator], maximum + math.log(denominator), trace


def evaluate(q, k, v, causal=False, query_offset=0, blocks=None):
    logits = score_rows(q, k)
    require(len(k) == len(v), 'K/V sequence mismatch')
    require(isinstance(query_offset, int) and query_offset >= 0, 'Nonnegative position offset required')
    blocks = blocks if blocks is not None else [list(range(0, min(2, len(k)))), list(range(2, len(k)))]
    outputs, traces = [], []
    for i, row in enumerate(logits):
        allowed = [not causal or j <= i + query_offset for j in range(len(k))]
        reference, ref_lse, probabilities = reference_row(row, v, allowed)
        output, lse, trace = online_row(row, v, blocks, allowed)
        check_close(output, reference)
        check_close(lse, ref_lse)
        check_close(math.fsum(probabilities), 1.0)
        outputs.append(output)
        traces.append(trace)
    return outputs, traces


def pretty(values):
    return '[' + ', '.join(f'{x:.12g}' for x in values) + ']'


def main():
    ln2 = math.log(2.0)
    q = [[math.sqrt(2.0) * ln2, 0.0], [0.0, math.sqrt(2.0) * ln2]]
    k = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    v = [[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]
    scores = score_rows(q, k)
    check_close(scores, [[ln2, 0.0, ln2], [0.0, ln2, ln2]])
    full, traces = evaluate(q, k, v)
    check_close(full, [[1.6, 2.0], [1.4, 2.4]])
    print('NONCAUSAL O:', '; '.join(pretty(row) for row in full))
    for i, trace in enumerate(traces):
        for step, state in enumerate(trace):
            print(f'ROW {i} BLOCK {step} keys={state["keys"]}: '
                  f'm={state["m"]:.12g} alpha={state["alpha"]:.12g} '
                  f'l={state["l"]:.12g} z={pretty(state["z"])}')
    causal, _ = evaluate(q, k, v, causal=True)
    check_close(causal, [[1.0, 0.0], [1.0 / 3, 4.0 / 3]])
    print('TOP-LEFT CAUSAL O:', '; '.join(pretty(row) for row in causal))
    shifted, _ = evaluate(q, k, v, causal=True, query_offset=1)
    check_close(shifted, [[2.0 / 3, 2.0 / 3], [1.4, 2.4]])
    print('OFFSET-1 CAUSAL O:', '; '.join(pretty(row) for row in shifted))
    # A new maximum must rescale the old numerator AND denominator.
    reordered, reordered_trace = evaluate(q, k, v, blocks=[[1], [0, 2]])
    check_close(reordered, full)
    check_close(reordered_trace[0][1]['alpha'], 0.5)
    print('RESCALE row0:', f'alpha={reordered_trace[0][1]["alpha"]:.12g}',
          f'l={reordered_trace[0][1]["l"]:.12g}',
          f'z={pretty(reordered_trace[0][1]["z"])}')
    for blocks in [[[0], [1], [2]], [[2, 0], [1]], [[0, 1, 2]], [[2], [1], [0]]]:
        check_close(evaluate(q, k, v, blocks=blocks)[0], full)
    for row in scores:
        shifted_logits = [x + 1000.0 for x in row]
        check_close(online_row(shifted_logits, v, [[0, 1], [2]])[0],
                    reference_row(row, v, [True] * 3)[0])
    # An initially empty masked tile is skipped, not exp(-inf - -inf).
    check_close(online_row(scores[0], v, [[2], [0, 1]], [True, False, False])[0], [1, 0])
    for operation in [lambda: online_row(scores[0], v, [[0, 1], [1, 2]]),
                      lambda: online_row(scores[0], v, [[0, 1], [2]], [False] * 3)]:
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid contract accepted')
    print('PASS: materialized/online references; causal positions; rescaling; partitions; shift stability; masked/invalid diagnostics')
    print('ATTENTION WALKTHROUGH PASS: CPU binary64 mathematics; no CUTLASS compilation or GPU execution')


if __name__ == '__main__':
    main()
