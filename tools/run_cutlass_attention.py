#!/usr/bin/env python3
"""Plan, or explicitly collect, pinned CUTLASS 88 Hopper FP16 forward evidence.

Independent of the book's existing workshop manifest. No GPU/build commands run
without --execute. An upstream zero exit is deliberately insufficient for PASS.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import shlex
import shutil
import subprocess
import sys
import tarfile

PIN = '1afc6d355b8cb992398f17beb4425702f1d57ddd'
EXAMPLE = 'examples/88_hopper_fmha'
TARGET = '88_hopper_fmha'
CASES = [
    ('d32-full', 1, 2, 128, 128, 32, 'no', 2),
    ('d64-full', 1, 2, 128, 128, 64, 'no', 3),
    ('d64-causal', 1, 2, 128, 128, 64, 'causal', 3),
    ('d128-full', 1, 2, 256, 256, 128, 'no', 2),
    ('d128-causal', 1, 2, 256, 256, 128, 'causal', 2),
    ('d256-full', 1, 2, 128, 128, 256, 'no', 1),
]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commands(output, jobs, iterations, backward=False):
    source, build = output / 'source', output / 'build'
    configure = ['cmake', '-S', str(source), '-B', str(build),
                 '-DCUTLASS_NVCC_ARCHS=90a', '-DCMAKE_BUILD_TYPE=Release',
                 '-DCUTLASS_ENABLE_TESTS=OFF', '-DCUTLASS_ENABLE_PROFILER=OFF',
                 '-DCUTLASS_ENABLE_EXAMPLES=ON', '-DCUTLASS_ENABLE_CUBLAS=OFF',
                 '-DCUTLASS_ENABLE_LIBRARY=OFF']
    compile_ = ['cmake', '--build', str(build), '--target', TARGET,
                '--parallel', str(jobs)]
    binary = build / EXAMPLE / TARGET
    runs = []
    for name, b, h, q, k, d, mask, count in CASES:
        runs.append((name, [str(binary), f'--b={b}', f'--h={h}', f'--q={q}',
                            f'--k={k}', f'--d={d}', f'--mask={mask}',
                            f'--iterations={iterations}', '--verify', '--verbose'], count))
    if backward:
        for mask in ['no', 'causal']:
            runs.append((f'd64-backward-{mask}', [str(binary), '--b=1', '--h=2', '--q=128', '--k=128', '--d=64', f'--mask={mask}', f'--iterations={iterations}', '--verify', '--verbose', '--bwd'], 2))
    return configure, compile_, runs


def classify(returncode, output, expected):
    """Classify the *entire* case, including every instantiated variant."""
    if returncode != 0:
        return 'FAIL'
    if re.search(r'failed (?:O|LSE|dQ|dK|dV):|Reference check failed|Failed to |Error running|returned an error:|cudaEvent\w*\(\) failed', output, re.I):
        return 'FAIL'
    if ('This example requires a GPU' in output or
            'No forward kernel instantiated' in output or
            'No backward kernel instantiated' in output or
            'Backward is not implemented' in output or
            'This kernel is not supported' in output):
        return 'SKIP'
    markers = re.findall(r'^\s*\[(OK|FAIL|--)\]\s+', output, re.M)
    if 'FAIL' in markers:
        return 'FAIL'
    # --verify is mandatory, and missing macro instantiations can emit no rows.
    if markers != ['OK'] * expected or not re.search(r' (?:Forward|Backward) ', output):
        return 'INCOMPLETE'
    return 'PASS'


def record(command, directory, label, timeout):
    started = utc()
    try:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=(os.name == 'posix'))
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            code = process.returncode
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            stdout, stderr = process.communicate()
            code = 124
            stderr += f'\nCollector timeout after {timeout} seconds; process group terminated.\n'
    except OSError as exc:
        code, stdout, stderr = 127, '', str(exc) + '\n'
    (directory / f'{label}.stdout.log').write_text(stdout)
    (directory / f'{label}.stderr.log').write_text(stderr)
    return {'argv': command, 'started_utc': started, 'finished_utc': utc(),
            'exit_code': code, 'stdout': f'{label}.stdout.log',
            'stderr': f'{label}.stderr.log',
            'stdout_sha256': sha(directory / f'{label}.stdout.log'),
            'stderr_sha256': sha(directory / f'{label}.stderr.log')}, stdout + '\n' + stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutlass', type=Path, required=True, help='local Git checkout containing the pinned revision')
    parser.add_argument('--output', type=Path, default=Path('artifacts/cutlass-attention'))
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=1800, help='seconds per configure/build/run command')
    parser.add_argument('--include-backward', action='store_true', help='also collect two FP16 d64 backward cases')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if min(args.jobs, args.iterations, args.timeout) <= 0:
        parser.error('jobs, iterations and timeout must be positive')
    output = args.output.resolve()
    configure, compile_, runs = commands(output, args.jobs, args.iterations, args.include_backward)
    revision = subprocess.run(['git', '-C', str(args.cutlass), 'rev-parse', f'{PIN}^{{commit}}'],
                              capture_output=True, text=True, check=False)
    if revision.returncode or revision.stdout.strip() != PIN:
        parser.error('checkout must contain the exact pinned v4.3.6 commit; run tools/fetch_cutlass.sh first')
    if not args.execute:
        blob = subprocess.check_output(['git', '-C', str(args.cutlass), 'show', f'{PIN}:{EXAMPLE}/88_hopper_fmha.cu'])
        print('PLAN METADATA: ' + json.dumps({'cutlass_commit': PIN, 'runner_source': f'{EXAMPLE}/88_hopper_fmha.cu', 'runner_source_sha256': hashlib.sha256(blob).hexdigest(), 'collector_sha256': sha(Path(__file__)), 'created_utc': utc(), 'include_backward': args.include_backward}))
        print(f'PLAN ONLY: CUTLASS {PIN}; no build or GPU commands executed.')
        for cmd in [configure, compile_] + [cmd for _, cmd, _ in runs]:
            print(shlex.join(cmd))
        return 0
    if output.exists():
        parser.error('output already exists; choose a new evidence directory')
    output.mkdir(parents=True)
    summary = {'schema': 1, 'started_utc': utc(), 'cutlass_commit': PIN,
               'scope': 'official example88 FP16 forward plus requested backward; vendor GPU reference, not a CPU/FP64 oracle',
               'status': 'INCOMPLETE', 'system': {'platform': platform.platform(),
               'machine': platform.machine(), 'python': sys.version,
               'environment': {k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'PATH', 'LD_LIBRARY_PATH']}},
               'records': [], 'cases': []}
    def save():
        summary['finished_utc'] = utc()
        (output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    try:
        # Git archive snapshots the immutable commit, never uncommitted local edits.
        archive = output / 'cutlass-source.tar'
        with archive.open('wb') as f:
            subprocess.run(['git', '-C', str(args.cutlass), 'archive', '--format=tar', PIN], stdout=f, check=True)
        summary['source_archive_sha256'] = sha(archive)
        source = output / 'source'
        source.mkdir()
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                dest = (source / member.name).resolve()
                if not dest.is_relative_to(source) or not (member.isfile() or member.isdir()):
                    raise ValueError(f'unsafe archive member: {member.name}')
            tf.extractall(source)
        snapshot = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
        (output / 'source-sha256.json').write_text(json.dumps(snapshot, indent=2) + '\n')
        shutil.copy2(__file__, output / 'collector.py')
        summary['collector_sha256'] = sha(output / 'collector.py')
        for label, cmd in [('gpu-system', ['nvidia-smi']), ('compiler-system', ['nvcc', '--version']),
                           ('cmake-system', ['cmake', '--version']), ('host-compiler-system', ['g++', '--version'])]:
            rec, _ = record(cmd, output, label, 30)
            summary['records'].append(rec)
        if shutil.which('nvcc') is None:
            summary.update(status='SKIP', reason='nvcc unavailable; no compilation or GPU validation')
            save()
            return 2
        for label, cmd in [('configure', configure), ('build', compile_)]:
            rec, _ = record(cmd, output, label, args.timeout)
            summary['records'].append(rec)
            save()
            if rec['exit_code'] != 0:
                summary.update(status='FAIL', reason=f'{label} failed')
                save()
                return 1
        for name, cmd, expected in runs:
            rec, text = record(cmd, output, name, args.timeout)
            rec.update(name=name, expected_verified_variants=expected,
                       status=classify(rec['exit_code'], text, expected))
            summary['cases'].append(rec)
            save()
        states = [case['status'] for case in summary['cases']]
        summary['status'] = ('PASS' if all(s == 'PASS' for s in states) else
                             'FAIL' if 'FAIL' in states else
                             'SKIP' if 'SKIP' in states else 'INCOMPLETE')
        save()
        print(f"{summary['status']}: {output / 'results.json'}")
        return 0 if summary['status'] == 'PASS' else 2
    except Exception as exc:
        summary.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
