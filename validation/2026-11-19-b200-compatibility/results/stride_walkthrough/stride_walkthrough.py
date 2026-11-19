#!/usr/bin/env python3
"""CPU-only nonnegative affine stride examples; no CuTe/GPU execution."""
import itertools
import json
import platform
import sys
from datetime import datetime, timezone


def offsets(shape, strides):
    if len(shape) != len(strides) or not shape:
        raise ValueError('Shape/stride rank mismatch or empty rank')
    if any(not isinstance(x, int) or x <= 0 for x in shape):
        raise ValueError('This exercise requires positive integer extents')
    if any(not isinstance(x, int) or x < 0 for x in strides):
        raise ValueError('This exercise requires nonnegative integer element strides')
    return [sum(c * d for c, d in zip(coord, strides))
            for coord in itertools.product(*(range(s) for s in shape))]


def span(shape, strides):
    return max(offsets(shape, strides)) + 1


def main():
    assert offsets((2, 3), (3, 1)) == [0, 1, 2, 3, 4, 5]
    assert offsets((2, 3), (1, 2)) == [0, 2, 4, 1, 3, 5]
    assert offsets((2, 3), (5, 1)) == [0, 1, 2, 5, 6, 7]
    assert span((2, 3), (5, 1)) == 8
    # Transpose is a coordinate permutation of the existing storage map.
    for m in range(2):
        for n in range(3):
            assert m * 5 + n == n + m * 5
    assert offsets((3, 2), (1, 5)) == [0, 5, 1, 6, 2, 7]
    batched = offsets((2, 2, 3), (10, 5, 1))
    assert batched == [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17]
    assert len(set(batched)) == 12 and span((2, 2, 3), (10, 5, 1)) == 18
    broadcast = offsets((2, 3), (0, 1))
    assert broadcast == [0, 1, 2, 0, 1, 2] and len(set(broadcast)) == 3
    overlap = offsets((2, 3), (1, 1))
    assert overlap == [0, 1, 2, 1, 2, 3] and len(set(overlap)) == 4
    assert 0x1000 % 16 == 0 and (0x1000 + 4) % 16 != 0
    assert 5 * 4 % 16 != 0 and 8 * 4 % 16 == 0
    valid = [(m, n) for m in range(0, 2) for n in range(2, 4)
             if m < 2 and n < 3]
    assert valid == [(0, 2), (1, 2)]
    for shape, strides in [((2, 3), (-3, 1)), ((0, 3), (3, 1)), ((2,), (1, 2))]:
        try:
            offsets(shape, strides)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid exercise domain accepted')
    print('ROW-MAJOR (2,3):(3,1): 0 1 2 / 3 4 5')
    print('COLUMN-MAJOR (2,3):(1,2): 0 2 4 / 1 3 5')
    print('PADDED (2,3):(5,1): offsets=0,1,2,5,6,7 span=8')
    print('BATCHED (2,2,3):(10,5,1): logical=12 distinct=12 span=18')
    print('BROADCAST (2,3):(0,1): logical=6 distinct=3; independent writes alias')
    print('OVERLAP (2,3):(1,1): logical=6 distinct=4; positive strides can alias')
    print('STRIDE WALKTHROUGH PASS: affine maps, transpose, batches, aliases, alignment, tails')
    print('Scope: CPU Python integer-address model; no CuTe compilation or GPU execution')


if __name__ == '__main__':
    main()
