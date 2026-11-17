#!/usr/bin/env python3
"""Emit a SHA-pinned Chapter 7 epilogue variant; opt in to Hopper execution with --run.

Default: Python-only pattern checks and an UNEXECUTED source/manifest in build/audit.
Archived sources, chapter listings, and historical validation records are never edited.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED = ROOT / 'docs/validation/2026-11-17/07-wgmma-part-2/ex01_wgmma_pipeline/ex01_wgmma_pipeline.cu'
SOURCE_SHA256 = '76960b5ead233cd05431fe9c4c7847f859e3270a7407461f4dbe91006c1c2790'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(f'Expected one replacement anchor, found {count}: {old[:70]!r}')
    return source.replace(old, new, 1)


def pattern_check():
    expected = []
    for row in range(64):
        for col in range(64):
            value = (2 ** (row // 16)) * (1 + (64 * (row % 16) + col) / 1024)
            dot = sum((2 ** (row // 16) if k == row % 16 else 0) *
                      (1 + (64 * k + col) / 1024) for k in range(16))
            if dot != value or struct.unpack('<e', struct.pack('<e', value))[0] != value:
                raise ValueError('Pattern is not an exact binary16 one-product dot.')
            expected.append(value)
    if len(set(expected)) != 4096:
        raise ValueError('Pattern must contain 4096 distinct outputs.')
    return dict(status='PASS', unique_values=len(set(expected)),
                exact_binary16=True, nonzero_products_per_dot=1,
                minimum=min(expected), maximum=max(expected))


def derive(raw):
    if sha(raw) != SOURCE_SHA256:
        raise ValueError('Archived source SHA mismatch; review anchors before updating the pin.')
    source = raw.decode('utf-8')
    original_kernel = source[:source.index('int main() {')]
    source = replace_once(source,
        '    for (int i = 0; i < A_N; i++) hA[i] = __float2half(0.01f);\n'
        '    for (int i = 0; i < B_N; i++) hB[i] = __float2half(0.01f);',
        '    // Audit-only exactly representable, unique-output inputs.\n'
        '    for (int r = 0; r < M; ++r)\n'
        '        for (int k = 0; k < K; ++k)\n'
        '            hA[r*K+k] = __float2half(k == r % 16 ? (float)(1 << (r/16)) : 0.0f);\n'
        '    for (int k = 0; k < K; ++k)\n'
        '        for (int c = 0; c < N; ++c)\n'
        '            hB[k*N+c] = __float2half(1.0f + (64*k+c)/1024.0f);')
    source = replace_once(source,
        '    printf("Expected: all acc values = K×0.01² = %.6f\\n\\n", K*0.01f*0.01f);',
        '    printf("AUDIT: 4096 unique exact FP16 outputs per tile; exact coordinate checks.\\n\\n");')
    source = replace_once(source,
        '    float rounded = __half2float(__float2half(0.01f));\n'
        '    float expected = K * rounded * rounded;\n'
        '    int errors = 0;\n'
        '    for (int i = 0; i < D_N; ++i) {\n'
        '        if (!std::isfinite(__half2float(hOA[i])) ||\n'
        '            fabsf(__half2float(hOA[i]) - expected) > 0.00002f) ++errors;\n'
        '        if (!std::isfinite(__half2float(hOB[i])) ||\n'
        '            fabsf(__half2float(hOB[i]) - expected) > 0.00002f) ++errors;\n'
        '    }',
        '    int errors = 0;\n'
        '    for (int i = 0; i < D_N; ++i) {\n'
        '        int r = i / N, c = i % N;\n'
        '        float expected = (float)(1 << (r/16)) *\n'
        '                         (1.0f + (64*(r%16)+c)/1024.0f);\n'
        '        float got_a = __half2float(hOA[i]), got_b = __half2float(hOB[i]);\n'
        '        if (got_a != expected) {\n'
        '            if (errors < 8) printf("A[%d,%d]: got %.9g expected %.9g\\n", r,c,got_a,expected);\n'
        '            ++errors;\n'
        '        }\n'
        '        if (got_b != expected) {\n'
        '            if (errors < 8) printf("B[%d,%d]: got %.9g expected %.9g\\n", r,c,got_b,expected);\n'
        '            ++errors;\n'
        '        }\n'
        '    }')
    if source[:source.index('int main() {')] != original_kernel:
        raise ValueError('Derivation modified the device implementation.')
    return source.encode('utf-8'), sha(original_kernel.encode('utf-8'))


def capture(command, directory, name, timeout):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        output = result.stdout + result.stderr
        code = result.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        output = str(error)
        code = None
    log = directory / (name + '.log')
    log.write_text(output)
    return dict(command=command, returncode=code, log=log.name,
                log_sha256=sha(log.read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Run preflight, compile and execute on Hopper.')
    args = parser.parse_args()
    raw = ARCHIVED.read_bytes()
    variant, kernel_sha = derive(raw)
    check = pattern_check()
    now = dt.datetime.now(dt.timezone.utc)
    directory = ROOT / 'build/audit' / (now.strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    source = directory / 'ch07_nonuniform_epilogue.cu'
    source.write_bytes(variant)
    report = dict(timestamp_utc=now.isoformat(), status='UNEXECUTED',
                  archived_source=str(ARCHIVED.relative_to(ROOT)), archived_sha256=sha(raw),
                  variant=source.name, variant_sha256=sha(variant),
                  unchanged_device_source_sha256=kernel_sha, replacement_count=3,
                  cpu_pattern_check=check,
                  scope='Opt-in audit variant; not historical validation or a performance benchmark.')
    manifest = directory / 'manifest.json'

    def save():
        manifest.write_text(json.dumps(report, indent=2) + '\n')

    save()
    if args.run:
        report['preflight'] = capture([sys.executable, str(ROOT / 'tools/h100_preflight.py'), '--json'],
                                      directory, 'preflight', 60)
        if report['preflight']['returncode'] != 0:
            report['status'] = 'BLOCKED'
        else:
            binary = directory / 'ch07_nonuniform_epilogue'
            report['compile'] = capture(['nvcc', '-std=c++17', '-arch=sm_90a', '-O2',
                                         str(source), '-o', str(binary)], directory, 'compile', 120)
            if report['compile']['returncode'] != 0:
                report['status'] = 'COMPILE_FAILED'
            else:
                report['binary_sha256'] = sha(binary.read_bytes())
                report['run'] = capture([str(binary)], directory, 'run', 60)
                report['status'] = 'PASSED' if report['run']['returncode'] == 0 else 'FAILED'
        report['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
    print(f"{report['status']}: {source}")
    print(f'Manifest: {manifest}')
    print('CPU: 4096 unique, exact binary16 one-product reference outputs; device source unchanged.')
    if not args.run:
        print('GPU variant has NOT been compiled or executed. Use --run on a Hopper host.')
    return 0 if report['status'] in {'UNEXECUTED', 'PASSED'} else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, struct.error) as error:
        sys.exit(f'Audit preparation failed: {error}')
