#!/usr/bin/env python3
"""Extract complete book programs into a fresh directory and retain audit evidence."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def programs(filters=()):
    # Keep this extraction heuristic identical to the original evidence collector.
    for page in sorted((ROOT / 'docs').glob('*.md')):
        if page.name.startswith('.'):
            continue
        source = page.read_text()
        for match in re.finditer(r'^```([^\n]*)\n(.*?)^```', source, re.M | re.S):
            code = match.group(2)
            if not (code.lstrip().startswith('//') and re.search(r'\bint\s+main\s*\(', code)):
                continue
            names = re.findall(r'\bex\w+\.cu\b', code)
            if not names:
                continue
            if filters and not any(value in page.name or value in names[0] for value in filters):
                continue
            yield dict(page=str(page.relative_to(ROOT)), line=source[:match.start()].count('\n') + 1,
                       example=names[0], sha256=hashlib.sha256(code.encode()).hexdigest(), code=code)


def execute(command, timeout):
    try:
        process = subprocess.run(command, capture_output=True, text=True, errors='replace', timeout=timeout)
        return process.returncode, process.stdout + process.stderr
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
        return 'timeout', decoded(error.stdout) + decoded(error.stderr) + f'\nTIMEOUT after {timeout}s\n'
    except OSError as error:
        return 'oserror', f'{type(error).__name__}: {error}\n'


def system_information(out):
    commands = [
        ['nvcc', '--version'], ['gcc', '--version'],
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'],
        ['nvidia-smi', 'topo', '-m'],
        [sys.executable, str(ROOT / 'tools/h100_preflight.py'), '--json'],
    ]
    lines = [f'UTC: {utc()}', f'OS: {platform.platform()}', f'Python: {platform.python_version()}']
    os_release = Path('/etc/os-release')
    if os_release.exists():
        lines.append(os_release.read_text())
    preflight = {}
    for command in commands:
        status, output = execute(command, 45)
        lines.extend([f'\n$ {shlex.join(command)}', f'Exit: {status}', output])
        if 'h100_preflight.py' in command[-2]:
            try:
                preflight = json.loads(output)
            except (ValueError, TypeError):
                pass
    (out / 'system-info.log').write_text('\n'.join(lines) + '\n')
    return preflight


def required_gpus(example):
    return 8 if example == 'ex13_hybrid_parallel_training.cu' else 2 if example in {
        'ex11_nvlink_bandwidth.cu', 'ex12_nccl_allreduce.cu'} else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filters', nargs='*', help='Substring filters for chapter or example names (OR).')
    parser.add_argument('--out', type=Path, help='Fresh output directory; a nonempty directory is rejected.')
    parser.add_argument('--list', action='store_true', help='Print source metadata as JSON; do not execute or create output.')
    parser.add_argument('--compile-only', action='store_true', help='Compile only; label every workload UNEXECUTED.')
    args = parser.parse_args(argv)
    selected = list(programs(args.filters))
    if args.list:
        print(json.dumps([{k: v for k, v in item.items() if k != 'code'} for item in selected], indent=2))
        return 0
    if not selected:
        parser.error('No complete programs match the filters.')
    out = args.out or ROOT / 'validation' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H%M%S.%fZ')
    out = out.expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f'Refusing to overwrite nonempty output: {out}')
    out.mkdir(parents=True, exist_ok=True)
    preflight = system_information(out)
    results, failed, blocked = [], False, False
    for item in selected:
        row = {k: v for k, v in item.items() if k != 'code'}
        row.update(timestamp=utc(), required_gpus=required_gpus(row['example']))
        folder = out / Path(row['page']).stem / Path(row['example']).stem
        folder.mkdir(parents=True)
        cu, binary = folder / row['example'], folder / 'example'
        cu.write_text(item['code'])
        command = ['nvcc', '-std=c++17', '-arch=sm_90a', '-O2', str(cu), '-o', str(binary)]
        if 'cudaLaunchDevice' in item['code'] or 'cudaGridDependencySynchronize' in item['code']:
            command += ['-rdc=true', '-lcudadevrt']
        if 'cuTensorMapEncode' in item['code']:
            command += ['-lcuda']
        if '#include <nccl.h>' in item['code']:
            command += ['-lnccl']
        if 'cublas' in item['code']:
            command += ['-lcublas', '-Xcompiler', '-pthread']
        row.update(command=command, run_command=[str(binary)], compile_timestamp=utc())
        row['compile_exit'], output = execute(command, 120)
        (folder / 'compile.log').write_text(output)
        row['run_exit'] = 'unexecuted'
        if row['compile_exit'] != 0:
            row['status'] = 'COMPILE_FAIL'
            failed = True
        elif args.compile_only:
            row['status'] = 'UNEXECUTED_COMPILE_ONLY'
        elif preflight.get('gpu_count', 0) < row['required_gpus']:
            row['status'] = 'BLOCKED_HARDWARE'
            blocked = True
            output = f"Requires {row['required_gpus']} visible GPU(s); discovered {preflight.get('gpu_count', 0)}. Workload not executed.\n"
            (folder / 'run.log').write_text(output)
        else:
            row['run_timestamp'] = utc()
            row['run_exit'], output = execute(row['run_command'], 90)
            (folder / 'run.log').write_text(output)
            # Some examples intentionally return zero after a prerequisite exit.
            prerequisite = re.search(r'(?:Need(?:s)?\s+(?:at least\s*|>=\s*)?\d+\s+GPUs?|no CUDA-capable device|no devices were found)', output, re.I)
            if prerequisite:
                row['status'] = 'BLOCKED_HARDWARE'
                blocked = True
            elif row['run_exit'] != 0:
                row['status'] = 'RUN_FAIL'
                failed = True
            else:
                row['status'] = 'PASS'
        row['finished_timestamp'] = utc()
        results.append(row)
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(row['page'], row['example'], row['status'], row['compile_exit'], row['run_exit'], flush=True)
    print(f'Evidence: {out}')
    return 1 if failed else 2 if blocked else 0


if __name__ == '__main__':
    sys.exit(main())
