#!/usr/bin/env python3
"""CPU reference for E2M1 nibbles and exact block scaling, not native NVFP4.

Python float scales are deliberately NOT rounded to E4M3, and there is no
per-tensor FP32 scaling stage or Blackwell scale-layout implementation.
No dependencies, GPU operations, timing claims or model training are involved.
"""
import math

MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode(code):
    if not isinstance(code, int) or not 0 <= code < 16:
        raise ValueError("E2M1 code must be an integer nibble")
    return math.copysign(MAGNITUDES[code & 7], -1.0 if code & 8 else 1.0)


def encode(value):
    """Nearest E2M1 value; ties to even code LSB; finite saturation at +/-6."""
    if not math.isfinite(value):
        raise ValueError("This reference accepts finite values only")
    sign = 8 if math.copysign(1.0, value) < 0 else 0
    magnitude = abs(value)
    if magnitude >= 6.0:
        return sign | 7
    index = min(range(8), key=lambda i: (abs(magnitude - MAGNITUDES[i]), i & 1))
    return sign | index


def pack(codes):
    if len(codes) % 2:
        raise ValueError("Explicitly pad an odd code count before byte packing")
    for code in codes:
        decode(code)  # Validate before shifting.
    return bytes(codes[i] | (codes[i + 1] << 4) for i in range(0, len(codes), 2))


def unpack(data):
    return [code for byte in data for code in (byte & 15, byte >> 4)]


def quantize_rows(rows, block_size=16):
    """Independent last-axis blocks with exact Python-float amax/6 scales."""
    if block_size < 1 or not rows or not rows[0]:
        raise ValueError("Nonempty rows and positive block size required")
    width = len(rows[0])
    if width % block_size or any(len(row) != width for row in rows):
        raise ValueError("Use equal rows with complete blocks")
    codes, scales, reconstructed = [], [], []
    for row in rows:
        if any(not math.isfinite(x) for x in row):
            raise ValueError("Finite input required")
        row_codes, row_scales, row_values = [], [], []
        for start in range(0, width, block_size):
            block = row[start:start + block_size]
            amax = max(abs(x) for x in block)
            scale = amax / 6.0 if amax else 1.0
            if scale == 0.0:
                raise ValueError("Scale underflow: rescale the input before quantizing")
            encoded = [encode(x / scale) for x in block]
            row_codes.extend(encoded)
            row_scales.append(scale)
            row_values.extend(decode(c) * scale for c in encoded)
        codes.append(row_codes)
        scales.append(row_scales)
        reconstructed.append(row_values)
    return codes, scales, reconstructed


def transpose(rows):
    return [list(column) for column in zip(*rows)]


def quantize_b_kn(b, block_size=16):
    """Logical B[K,N]: block along K, by quantizing rows of B transpose."""
    if not b or not b[0] or any(len(row) != len(b[0]) for row in b):
        raise ValueError("Rectangular nonempty B required")
    codes_nk, scales_nk, values_nk = quantize_rows(transpose(b), block_size)
    return codes_nk, scales_nk, transpose(values_nk)


def run_checks():
    # Every nibble, including both signed zeros, survives decode/encode.
    for code in range(16):
        assert encode(decode(code)) == code
    assert math.copysign(1.0, decode(8)) < 0
    # All 256 ordered nibble pairs: low nibble first, high nibble second.
    for low in range(16):
        for high in range(16):
            assert unpack(pack([low, high])) == [low, high]
    for i in range(7):
        middle = (MAGNITUDES[i] + MAGNITUDES[i + 1]) / 2
        even = i if i % 2 == 0 else i + 1
        assert encode(middle) == even
        assert encode(-middle) == (even | 8)
    assert decode(encode(100.0)) == 6.0
    assert decode(encode(-100.0)) == -6.0
    assert decode(encode(1e308)) == 6.0
    _, scales, values = quantize_rows([[0.0] * 16, [-0.0] * 16])
    assert scales == [[1.0], [1.0]]
    assert all(x == 0.0 for row in values for x in row)
    assert all(math.copysign(1.0, x) < 0 for x in values[1])
    # E2M1-representable values at two exact scales: no quantization error.
    row = list(MAGNITUDES) * 2 + [x * 8 for x in MAGNITUDES] * 2
    codes, scales, values = quantize_rows([row])
    assert scales == [[1.0, 8.0]] and values == [row]
    assert unpack(pack(codes[0])) == codes[0]
    # Deliberately asymmetric B: K blocks must follow each column, not N rows.
    b = [[float((k % 8) - 4), float(8 * ((k % 8) - 4))] for k in range(32)]
    b_codes, b_scales, restored = quantize_b_kn(b)
    assert b_scales == [[4.0 / 6, 4.0 / 6], [32.0 / 6, 32.0 / 6]]
    for n in range(2):
        for k in range(32):
            assert restored[k][n] == decode(b_codes[n][k]) * b_scales[n][k // 16]
    # Invalid inputs and incomplete blocks are diagnosed, not silently padded.
    for operation in (lambda: encode(float('nan')), lambda: encode(float('inf')),
                      lambda: pack([0]), lambda: decode(16),
                      lambda: quantize_rows([[1.0] * 15]),
                      lambda: quantize_b_kn([[1.0], [1.0, 2.0]])):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("Expected an invalid-input diagnosis")
    print("PASS: all 16 E2M1 codes, 256 nibble pairs and nearest-even ties")
    print("PASS: finite saturation, signed zero, zero blocks and exact-scale round trips")
    print("PASS: K-axis B scaling, independent block scales and invalid-input diagnoses")
    print("Scope: CPU format/layout reference; no E4M3 scale quantization, tensor scale or native NVFP4 GEMM")
    print("FP4 REFERENCE PASS: 3 CPU-only reference groups")


if __name__ == '__main__':
    run_checks()
