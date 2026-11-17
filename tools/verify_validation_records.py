#!/usr/bin/env python3
"""Reject missing, changed or unsupported dated CUDA execution records."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def verify(root, date):
    public = root / 'docs' / 'validation' / date
    rows = json.loads((public / 'results.json').read_text())
    recorded, complete, errors = {}, {}, []
    preflight = root / 'docs' / 'assets' / 'tools' / 'h100_preflight.py'
    if not preflight.is_file() or preflight.read_bytes() != (root / 'tools' / 'h100_preflight.py').read_bytes():
        errors.append('Standalone preflight download differs from the repository utility.')
    for page in sorted((root / 'docs').glob('*.md')):
        if page.name.startswith('.'):
            continue
        for block in re.finditer(r'^```[^\n]*\n(.*?)^```', page.read_text(), re.M | re.S):
            code = block[1]
            names = re.findall(r'\bex\w+\.cu\b', code)
            if not (names and code.lstrip().startswith('//') and re.search(r'\bint\s+main\s*\(', code)):
                continue
            key = (str(page.relative_to(root)), names[0])
            if key in complete:
                errors.append(f'Duplicate program: {key}')
            complete[key] = digest(code)
    passed, blocked = 0, 0
    for row in rows:
        key = row['page'], row['example']
        if key in recorded:
            errors.append(f'Duplicate record: {key}')
        recorded[key] = row['sha256']
        folder = public / Path(row['page']).stem / Path(row['example']).stem
        source = folder / row['example']
        if not source.is_file() or digest(source.read_text()) != row['sha256']:
            errors.append(f'Downloadable source does not match execution record: {key}')
        if complete.get(key) != row['sha256']:
            errors.append(f'Chapter source does not match execution record: {key}')
        if row.get('compile_exit') != 0:
            errors.append(f'Compilation did not pass: {key}')
        for name in ('compile.log', 'run.log'):
            if not (folder / name).is_file():
                errors.append(f'Missing {name}: {key}')
        run = (folder / 'run.log').read_text() if (folder / 'run.log').is_file() else ''
        needs_hardware = 'Need at least 2 GPUs' in run or 'Need >= 2 GPUs' in run or 'Need 8 GPUs' in run
        if needs_hardware:
            blocked += 1
            if 'multi-GPU workload NOT RUN' not in (root / row['page']).read_text():
                errors.append(f'Missing hardware limitation notice: {key}')
        elif row.get('run_exit') != 0:
            errors.append(f'Execution did not pass: {key}')
        else:
            passed += 1
    for key in complete.keys() - recorded.keys():
        errors.append(f'Complete program has no dated execution record: {key}')
    for message in errors:
        print('FAIL:', message)
    if errors:
        return 1
    print(f'PASS: {len(rows)} complete programs match their dated source hashes.')
    print(f'Coverage: {passed} successful workloads; {blocked} hardware-blocked workloads, explicitly labeled.')
    print('Scope: evidence consistency only; this command does not execute CUDA or prove numerical correctness.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--date', default='2026-11-17')
    args = parser.parse_args()
    try:
        return verify(args.root.resolve(), args.date)
    except (OSError, ValueError, KeyError) as error:
        print('FAIL:', error)
        return 1


if __name__ == '__main__':
    sys.exit(main())
