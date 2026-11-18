#!/usr/bin/env python3
"""Verify fresh published evidence without executing CUDA or rewriting history."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from validate_examples import ROOT, programs

NAME = '2026-11-17-final-audit'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, public=None, summary=None):
    public = public or root / 'docs/validation' / NAME
    summary = summary or root / 'docs/final-audit.md'
    errors = []
    inventory = json.loads((public / 'integrity.json').read_text())
    actual_files = {str(p.relative_to(public)) for p in public.rglob('*') if p.is_file()} - {'integrity.json'}
    if actual_files != set(inventory['files']):
        errors.append('Published file inventory differs from integrity record.')
    for name, expected in inventory['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            errors.append(f'Unsafe inventory path: {name}')
        elif not (public / relative).is_file() or sha(public / relative) != expected:
            errors.append(f'Changed or missing evidence: {name}')
    if sha(summary) != inventory['summary_sha256']:
        errors.append('Published summary differs from verified inventory.')
    # programs() intentionally uses the identical collector extraction heuristic.
    current = {(x['page'], x['example']): x['sha256'] for x in programs()}
    rows = json.loads((public / 'results.json').read_text())
    recorded = {(x['page'], x['example']): x['sha256'] for x in rows}
    expected_current = dict(recorded)
    # Only this portable training program may be superseded by the separately
    # verified eight-A100 PASS. Every other archived source remains exact.
    from verify_eight_gpu_audit import KEY, superseding_hash
    if current.get(KEY) != recorded.get(KEY):
        try:
            expected_current[KEY] = superseding_hash(root)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append('Eight-A100 supplement cannot supersede ex13: ' + str(error))
    if len(rows) != 27 or len(recorded) != 27 or current != expected_current:
        errors.append('Fresh audit must cover all 27 hashes, with only verified eight-A100 ex13 supersession allowed.')
    passed = blocked = 0
    for row in rows:
        key = row['example']
        folder = public / Path(row['page']).stem / Path(key).stem
        if sha(folder / key) != row['sha256']:
            errors.append(f'Downloaded source mismatch: {key}')
        if row.get('compile_exit') != 0:
            errors.append(f'Compilation did not pass: {key}')
        for field in ('timestamp', 'compile_timestamp', 'finished_timestamp'):
            if not re.match(r'^\d{4}-\d\d-\d\dT.*(?:\+00:00|Z)$', row.get(field, '')):
                errors.append(f'Missing UTC {field}: {key}')
        if not row.get('command') or not row.get('run_command'):
            errors.append(f'Missing captured commands: {key}')
        output = (folder / 'run.log').read_text()
        prerequisite = re.search(r'Need(?:s)?\s+(?:at least\s*|>=\s*)?\d+\s+GPUs?|Requires \d+ visible GPU', output, re.I)
        if row.get('status') == 'PASS':
            passed += 1
            if row.get('run_exit') != 0 or prerequisite:
                errors.append(f'PASS contradicts execution: {key}')
        elif row.get('status') == 'BLOCKED_HARDWARE':
            blocked += 1
            if not prerequisite:
                errors.append(f'Blocked status lacks prerequisite diagnostic: {key}')
        else:
            errors.append(f'Unsuccessful or unexecuted workload: {key}')
        if row['timestamp'] not in summary.read_text() or row['sha256'] not in summary.read_text():
            errors.append(f'Summary omits execution metadata: {key}')
    if not (public / 'system-info.log').read_text().strip():
        errors.append('Empty system information.')
    manifest = json.loads((public / 'nonuniform/manifest.json').read_text())
    if manifest.get('status') != 'PASSED':
        errors.append('Nonuniform audit did not pass.')
    variant = public / 'nonuniform' / manifest['variant']
    if sha(variant) != manifest['variant_sha256']:
        errors.append('Nonuniform source hash mismatch.')
    archive = root / manifest['archived_source']
    if sha(archive) != manifest['archived_sha256']:
        errors.append('Nonuniform historical baseline hash mismatch.')
    device = variant.read_text().split('int main() {')[0].encode()
    if hashlib.sha256(device).hexdigest() != manifest['unchanged_device_source_sha256']:
        errors.append('Nonuniform device implementation changed.')
    for stage in ('preflight', 'compile', 'run'):
        item = manifest[stage]
        if item.get('returncode') != 0 or sha(public / 'nonuniform' / item['log']) != item['log_sha256']:
            errors.append(f'Nonuniform {stage} evidence mismatch.')
    memcheck = public / 'nonuniform/memcheck.log'
    memcheck_claim = 'ERROR SUMMARY: 0 errors' in summary.read_text()
    if memcheck.exists():
        if not re.search(r'^=+\s*ERROR SUMMARY: 0 errors\s*$', memcheck.read_text(), re.M):
            errors.append('Published memcheck log does not report zero errors.')
        if not memcheck_claim:
            errors.append('Published memcheck log is absent from summary.')
    elif memcheck_claim:
        errors.append('Summary claims memcheck without a captured log.')
    attention = (public / 'attention/run.log').read_text()
    if len(re.findall(r'^N=\d+ d=\d+ causal=(?:True|False): PASS ', attention, re.M)) != 12 or 'PASS: 12 cases, including causal masks and partial final tiles.' not in attention:
        errors.append('Attention log lacks 12 successful cases.')
    attention_source = public / 'attention/validate_attention.py'
    if sha(attention_source) != inventory['attention_source_sha256'] or attention_source.read_bytes() != (root / 'tools/validate_attention.py').read_bytes():
        errors.append('Attention executed source differs from current reference.')
    text = summary.read_text()
    if f'**{passed} PASS**' not in text or f'**{blocked} hardware-blocked**' not in text:
        errors.append('Summary workload counts contradict records.')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        errors = verify(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors = [str(error)]
    for error in errors:
        print('FAIL:', error)
    if not errors:
        print('PASS: fresh audit source hashes, evidence inventory, statuses, logs and summary agree.')
        print('Scope: evidence consistency; CUDA is not executed by this verifier.')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
