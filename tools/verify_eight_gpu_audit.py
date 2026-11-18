#!/usr/bin/env python3
"""Verify the separate eight-A100 training supplement without running CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import re

NAME = '2026-11-18-eight-a100'
KEY = ('docs/11-multi-gpu-nccl.md', 'ex13_hybrid_parallel_training.cu')
MARKER = 'All-rank training validation: PASS'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_source(root):
    text = (root / KEY[0]).read_text()
    found = [m[1] for m in re.finditer(r'^```[^\n]*\n(.*?)^```', text, re.M | re.S)
             if m[1].lstrip().startswith('//') and KEY[1] in m[1] and re.search(r'\bint\s+main\s*\(', m[1])]
    if len(found) != 1:
        raise ValueError('Expected exactly one current ex13 source.')
    return found[0]


def check_data(folder):
    errors = []
    m = json.loads((folder / 'manifest.json').read_text())
    if m.get('source') != KEY[1] or m.get('arch') != 'sm_80' or m.get('status') != 'PASS' or m.get('pass_marker') != MARKER:
        errors.append('Supplement must be a PASS for exact ex13, sm_80 and all-rank marker.')
    if sha(folder / KEY[1]) != m.get('source_sha256'):
        errors.append('Supplement source hash mismatch.')
    if sha(folder / 'system-info.log') != m.get('system_sha256'):
        errors.append('Supplement system information hash mismatch.')
    system = (folder / 'system-info.log').read_text()
    gpu_lines = [line for line in system.splitlines() if re.search(r'A100.*GPU-', line)]
    if len(gpu_lines) != 8 or any('8.0' not in line for line in gpu_lines):
        errors.append('System information must identify eight A100 compute-capability 8.0 devices.')
    if 'nvidia-smi topo -m' not in system:
        errors.append('Missing captured topology.')
    for field in ('timestamp_utc', 'finished_utc'):
        if not re.fullmatch(r'\d{4}-\d\d-\d\dT.*(?:\+00:00|Z)', m.get(field, '')):
            errors.append('Missing UTC ' + field)
    for stage in ('compile', 'run'):
        item = m.get(stage, {})
        if item.get('exit_code') != 0 or item.get('timed_out') is not False or not item.get('command'):
            errors.append(stage + ' must have a successful captured command and exit.')
        for field in ('started_utc', 'finished_utc'):
            if not re.fullmatch(r'\d{4}-\d\d-\d\dT.*(?:\+00:00|Z)', item.get(field, '')):
                errors.append(stage + ' lacks UTC ' + field)
        name = item.get('log', '')
        if name != stage + '.log' or sha(folder / (stage + '.log')) != item.get('log_sha256'):
            errors.append(stage + ' log hash/name mismatch.')
    command = m.get('compile', {}).get('command', [])
    if '-arch=sm_80' not in command or not any(str(x).endswith('/' + KEY[1]) for x in command):
        errors.append('Captured compiler command does not target the stated source/architecture.')
    run = (folder / 'run.log').read_text()
    if MARKER not in run or re.search(r'All-rank training validation: FAIL|\[Verify\].*FAIL|Need 8 GPUs|Requires 8 visible', run):
        errors.append('Run lacks successful all-rank validation or contains failed verification.')
    if len(re.findall(r'\[Verify\] GPU \d+ vs GPU \d+ \(DP replica diff\): max_diff = .*PASS', run)) != 4:
        errors.append('Run must contain four successful DP replica-pair comparisons.')
    weights = re.findall(r'\[Verify\] GPU (\d+) CPU final-weight reference: PASS; elements=2048(?: |$)', run)
    if len(weights) != 8 or set(weights) != set(map(str, range(8))):
        errors.append('Run must contain all eight successful CPU final-weight references.')
    losses = re.findall(r'\[Verify\] DP replica (\d+) per-step CPU loss reference: PASS; steps=8(?: |$)', run)
    if len(losses) != 2 or set(losses) != {'0', '1'}:
        errors.append('Run must contain both successful per-step CPU loss references.')
    return errors


def verify(root, public=None, summary=None):
    public = public or root / 'docs/validation' / NAME
    summary = summary or root / 'docs/eight-gpu-validation.md'
    errors = check_data(public)
    m = json.loads((public / 'manifest.json').read_text())
    if hashlib.sha256(current_source(root).encode()).hexdigest() != m['source_sha256']:
        errors.append('Current chapter ex13 differs from eight-A100 executed source.')
    inventory = json.loads((public / 'integrity.json').read_text())
    expected = {'manifest.json', KEY[1], 'compile.log', 'run.log', 'system-info.log'}
    actual = {str(p.relative_to(public)) for p in public.rglob('*') if p.is_file()} - {'integrity.json'}
    if actual != expected or set(inventory.get('files', {})) != expected:
        errors.append('Supplement inventory must contain exactly source, manifest and three logs.')
    for name in expected:
        if inventory.get('files', {}).get(name) != sha(public / name):
            errors.append('Supplement inventory mismatch: ' + name)
    if sha(summary) != inventory.get('summary_sha256'):
        errors.append('Supplement summary hash mismatch.')
    text = summary.read_text()
    for value in (m['source_sha256'], m['timestamp_utc'], 'A100', 'sm_80', MARKER):
        if value not in text:
            errors.append('Supplement summary omits ' + value)
    return errors


def superseding_hash(root):
    errors = verify(root)
    if errors:
        raise ValueError('; '.join(errors))
    return json.loads((root / 'docs/validation' / NAME / 'manifest.json').read_text())['source_sha256']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    try:
        errors = verify(a.root.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors = [str(error)]
    for error in errors:
        print('FAIL:', error)
    if not errors:
        print('PASS: separate eight-A100 source, logs, status, hardware and summary agree.')
    return bool(errors)

if __name__ == '__main__':
    raise SystemExit(main())
