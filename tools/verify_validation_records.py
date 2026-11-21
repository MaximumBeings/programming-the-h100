#!/usr/bin/env python3
"""Reject missing, changed or unsupported dated CUDA execution records."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
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
    from review_revisions import historical_hashes
    complete = historical_hashes(root, complete)
    current_only = {('docs/11-multi-gpu-nccl.md', name) for name in (
        'ex11_nvlink_bandwidth.cu', 'ex12_nccl_allreduce.cu', 'ex13_hybrid_parallel_training.cu')}
    from pending_revalidation import pending
    revisions = pending(root, complete, {(row['page'], row['example']): row['sha256'] for row in rows})
    passed, blocked = 0, 0
    fresh_rows = None
    superseded = set()
    for row in rows:
        key = row['page'], row['example']
        if key in recorded:
            errors.append(f'Duplicate record: {key}')
        recorded[key] = row['sha256']
        folder = public / Path(row['page']).stem / Path(row['example']).stem
        source = folder / row['example']
        if not source.is_file() or digest(source.read_text()) != row['sha256']:
            errors.append(f'Downloadable source does not match execution record: {key}')
        if complete.get(key) != row['sha256'] and key not in revisions:
            # Historical downloads stay immutable. A changed chapter source needs
            # a fully verified fresh PASS record and an explicit displayed notice.
            if fresh_rows is None:
                fresh_rows = {}
                try:
                    from verify_final_audit import verify as verify_fresh
                    fresh_errors = verify_fresh(root)
                    if not fresh_errors:
                        fresh = json.loads((root / 'docs/validation/2026-11-17-final-audit/results.json').read_text())
                        fresh_rows = {(r['page'], r['example']): r for r in fresh}
                        from verify_eight_gpu_audit import KEY, superseding_hash
                        if complete.get(KEY) != fresh_rows.get(KEY, {}).get('sha256'):
                            fresh_rows[KEY] = dict(sha256=superseding_hash(root), compile_exit=0,
                                                   run_exit=0, status='PASS')
                    else:
                        errors.append('Fresh evidence cannot supersede history: ' + '; '.join(fresh_errors))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    errors.append(f'Fresh evidence cannot supersede history: {error}')
            newer = fresh_rows.get(key, {})
            if (newer.get('sha256') != complete.get(key) or newer.get('compile_exit') != 0
                    or newer.get('status') != 'PASS' or newer.get('run_exit') != 0):
                errors.append(f'Chapter source has no verified fresh PASS record: {key}')
            else:
                superseded.add(key)
        if row.get('compile_exit') != 0:
            errors.append(f'Compilation did not pass: {key}')
        for name in ('compile.log', 'run.log'):
            if not (folder / name).is_file():
                errors.append(f'Missing {name}: {key}')
        run = (folder / 'run.log').read_text() if (folder / 'run.log').is_file() else ''
        needs_hardware = 'Need at least 2 GPUs' in run or 'Need >= 2 GPUs' in run or 'Need 8 GPUs' in run
        chapter = (root / row['page']).read_text()
        marker = re.compile(r'<!-- validation:start:' + re.escape(row['example']) +
                            r' -->.*?<!-- validation:end:' + re.escape(row['example']) + r' -->', re.S)
        sections = marker.findall(chapter)
        undisplayed = not sections and key in current_only and key in superseded
        if undisplayed:
            # Narrow exemption: old raw archives remain checked above; the exact
            # current source needs verified PASS evidence and its full displayed log.
            if key[1] == 'ex13_hybrid_parallel_training.cu':
                from verify_eight_gpu_audit import NAME, verify as verify_extra
                errors.extend(verify_extra(root))
                current_folder = root / 'docs/validation' / NAME
                current_meta = json.loads((current_folder / 'manifest.json').read_text())
                current_time = current_meta['timestamp_utc']
                current_command = current_meta['compile']['command']
            else:
                current_folder = root / 'docs/validation/2026-11-17-final-audit' / Path(key[0]).stem / Path(key[1]).stem
                current_time = fresh_rows[key]['run_timestamp']
                current_command = fresh_rows[key]['command']
            normalise_current = lambda value: '\n'.join(line.rstrip() for line in value.splitlines()).strip('\n')
            current_output = normalise_current((current_folder / 'run.log').read_text())
            shown_command = [key[1] if str(arg).endswith('/' + key[1]) else Path(key[1]).stem if str(arg).endswith('/example') else arg for arg in current_command]
            if shlex.join(shown_command) not in chapter:
                errors.append(f'Current captured compiler command missing or different: {key}')
            if current_output not in normalise_current(chapter) or current_time not in chapter or str(current_folder.relative_to(root / 'docs')) not in chapter:
                errors.append(f'Current verified PASS capture must replace historical display: {key}')
        elif len(sections) != 1:
            errors.append(f'Expected one displayed execution record: {key}')
        else:
            section = sections[0]
            if key in superseded and not ('historical' in section.lower() and 'supersed' in section.lower() and ('final-audit.md' in section or ('eight-gpu-validation.md' in section and key == ('docs/11-multi-gpu-nccl.md', 'ex13_hybrid_parallel_training.cu')))):
                errors.append(f'Changed source needs a historical/superseded notice and fresh audit link inside its displayed record: {key}')
            shown = re.search(r'```text\n(.*?)\n```', section, re.S)
            normalise = lambda value: '\n'.join(line.rstrip() for line in value.splitlines()).strip('\n')
            if not shown or normalise(shown[1]) != normalise(run):
                errors.append(f'Displayed output differs from captured run.log: {key}')
            if row.get('timestamp', '') not in section:
                errors.append(f'Displayed execution date differs from the record: {key}')
            expected_command = [row['example'] if arg.endswith('/' + row['example'])
                                else Path(row['example']).stem if arg.endswith('/example')
                                else arg for arg in row['command']]
            command = re.search(r'```bash\n(.*?)\n```', section, re.S)
            expected_shell = shlex.join(expected_command) + '\n./' + Path(row['example']).stem
            if not command or command[1].strip() != expected_shell:
                errors.append(f'Displayed compile/run commands differ from captured command: {key}')
            expected_status = 'multi-GPU workload NOT RUN' if needs_hardware else 'execution PASS'
            if expected_status not in section:
                errors.append(f'Displayed workload status contradicts the captured run: {key}')
        if needs_hardware:
            blocked += 1
            if not undisplayed and 'multi-GPU workload NOT RUN' not in (root / row['page']).read_text():
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
    print(f'PASS: {len(rows)} archive records match their downloads and logs; historical displays match captured executions, including {len(superseded)} superseding PASS records.')
    from review_revisions import revisions as review_pending
    pending_count = sum(r['status'] == 'pending_gpu_revalidation' for r in revisions.values()) + sum('source' not in r for r in review_pending(root))
    print(f'Current coverage: {27 - pending_count} verified revisions; {pending_count} awaiting GPU revalidation.')
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
