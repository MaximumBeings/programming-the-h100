#!/usr/bin/env python3
"""Publish a separate fresh audit without rewriting historical chapter evidence."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
import shlex
import shutil
import tempfile
from validate_examples import ROOT, programs

NAME = '2026-11-17-final-audit'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normal(command, example):
    return shlex.join([example if arg.endswith('/' + example) else 'example' if arg.endswith('/example') else arg for arg in command])


def build(source, nonuniform, attention, attention_source, destination, report):
    rows = json.loads((source / 'results.json').read_text())
    current = {(x['page'], x['example']): x['sha256'] for x in programs()}
    recorded = {(x['page'], x['example']): x['sha256'] for x in rows}
    if len(rows) != 27 or len(recorded) != 27 or current != recorded:
        raise ValueError('Require all 27 current complete sources, exactly once.')
    if any(row.get('compile_exit') != 0 for row in rows):
        raise ValueError('Fresh audit includes an unsuccessful compilation.')
    counts = collections.Counter(row.get('status') for row in rows)
    if set(counts) - {'PASS', 'BLOCKED_HARDWARE'}:
        raise ValueError(f'Audit is incomplete or failed: {dict(counts)}')
    if not attention_source.is_file() or attention_source.read_bytes() != (ROOT / 'tools/validate_attention.py').read_bytes():
        raise ValueError('Remote attention source is missing or differs from current source.')
    if 'PASS: 12 cases, including causal masks and partial final tiles.' not in attention.read_text():
        raise ValueError('Attention log does not record all 12 passing cases.')
    manifest = json.loads((nonuniform / 'manifest.json').read_text())
    if manifest.get('status') != 'PASSED':
        raise ValueError('Nonuniform audit has not passed.')
    destination.mkdir(parents=True)
    def copy(src, relative):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    copy(source / 'results.json', Path('results.json'))
    copy(source / 'system-info.log', Path('system-info.log'))
    for row in rows:
        folder = Path(Path(row['page']).stem) / Path(row['example']).stem
        for filename in (row['example'], 'compile.log', 'run.log'):
            copy(source / folder / filename, folder / filename)
        if sha(destination / folder / row['example']) != row['sha256']:
            raise ValueError(f"Source hash mismatch: {row['example']}")
    for filename in ['manifest.json', manifest['variant']] + [manifest[key]['log'] for key in ('preflight', 'compile', 'run')]:
        if Path(filename).name != filename:
            raise ValueError('Nonuniform filenames must be plain basenames.')
        copy(nonuniform / filename, Path('nonuniform') / filename)
    memcheck = nonuniform / 'memcheck.log'
    if memcheck.exists():
        if not re.search(r'^=+\s*ERROR SUMMARY: 0 errors\s*$', memcheck.read_text(), re.M):
            raise ValueError('Optional memcheck log does not report ERROR SUMMARY: 0 errors.')
        copy(memcheck, Path('nonuniform/memcheck.log'))
    copy(attention, Path('attention/run.log'))
    copy(attention_source, Path('attention/validate_attention.py'))
    base = f'validation/{NAME}'
    system_lines = (source / 'system-info.log').read_text().splitlines()
    selected_system = [line for line in system_lines if line.startswith(('UTC:', 'OS:', 'Python:', 'PRETTY_NAME=', 'Cuda compilation tools,', 'gcc (', 'NVIDIA H100'))]
    system_excerpt = '\n'.join(selected_system)
    text = [
        '# Final audit — two H100 GPUs', '',
        'This fresh audit supplements the [original November 17 evidence](validation/2026-11-17/results.json). Historical records and chapter transcripts are preserved.', '',
        f'All **{len(rows)} complete CUDA programs compiled**. Workload statuses: **{counts["PASS"]} PASS**, **{counts["BLOCKED_HARDWARE"]} hardware-blocked**. A prerequisite exit is not a successful workload.', '',
        f'[System and tool information]({base}/system-info.log) · [Raw execution manifest]({base}/results.json) · [Integrity inventory]({base}/integrity.json). The system log records CUDA-visible discovery, GPU names, memory, driver, compiler, OS and topology.', '',
        'Each command below is normalized to a local source filename and `example` executable. The raw manifest preserves the actual host paths, commands, stage exits and UTC timestamps. Run commands belong in the matching extracted source directory.', '',
        '## Execution system', '', '```text', system_excerpt, '```', '',
        '## Complete program results', '',
    ]
    for row in rows:
        folder = f"{base}/{Path(row['page']).stem}/{Path(row['example']).stem}"
        text += [f'### {row["example"]}', '',
                 f'**{row["status"]}** · UTC: `{row["timestamp"]}` · compile exit: `{row["compile_exit"]}` · run exit: `{row["run_exit"]}`.', '',
                 f'Compile UTC: `{row.get("compile_timestamp", "unrecorded")}` · Run UTC: `{row.get("run_timestamp", "not executed")}` · Finished UTC: `{row.get("finished_timestamp", "unrecorded")}`.', '',
                 f'Source SHA-256: `{row["sha256"]}`.', '',
                 f'[Source]({folder}/{row["example"]}) · [Compiler output]({folder}/compile.log) · [Runtime output]({folder}/run.log).', '',
                 '```bash', normal(row['command'], row['example']), './example', '```', '']
    text += ['## Nonuniform epilogue audit', '',
             f'Status: **{manifest["status"]}** · UTC: `{manifest["timestamp_utc"]}`. This opt-in variant uses 4,096 unique exactly representable expected outputs and preserves the device implementation. It strengthens layout coverage; it is not a new performance benchmark.', '',
             f'[Manifest]({base}/nonuniform/manifest.json) · [Variant source]({base}/nonuniform/{manifest["variant"]}) · [Compiler output]({base}/nonuniform/{manifest["compile"]["log"]}) · [Runtime output]({base}/nonuniform/{manifest["run"]["log"]}).', '',
             '## Online-softmax numerical reference', '',
             f'All **12 cases passed** against PyTorch math SDPA, including causal masks and partial final tiles. Execution metadata is retained verbatim in the captured log. [Captured output]({base}/attention/run.log) · [Executed reference source]({base}/attention/validate_attention.py). This verifies the recurrence for those cases, not the FA3 skeleton’s synchronization or published throughput.', '',
             '## Verification scope', '',
             'CI verifies source and evidence hashes, workload status consistency, captured-log presence and this summary. CI does not rerun CUDA or establish untested numerical coverage. The remaining eight-GPU workload requires the corresponding hardware.', '']
    if memcheck.exists():
        position = text.index('## Online-softmax numerical reference')
        text[position:position] = [f'Compute Sanitizer memcheck of this nonuniform variant reports **ERROR SUMMARY: 0 errors**. [Captured memcheck log]({base}/nonuniform/memcheck.log). This is memory-access checking for that execution, not proof of every asynchronous protocol or a performance measurement.', '']
    report.write_text('\n'.join(text))
    inventory = {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    integrity = dict(files=inventory, summary_sha256=sha(report), attention_source_sha256=sha(attention_source))
    (destination / 'integrity.json').write_text(json.dumps(integrity, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'validation/final-audit-two-h100')
    parser.add_argument('--nonuniform', type=Path, default=ROOT / 'validation/final-nonuniform')
    parser.add_argument('--attention', type=Path, default=ROOT / 'validation/final-attention.log')
    parser.add_argument('--attention-source', type=Path, default=ROOT / 'validation/final-attention.py')
    args = parser.parse_args()
    destination, summary = ROOT / 'docs/validation' / NAME, ROOT / 'docs/final-audit.md'
    if destination.exists() or summary.exists():
        parser.error('Final audit already exists; refusing to overwrite published evidence.')
    # Prepare everything in ignored scratch space before publishing any result.
    with tempfile.TemporaryDirectory(prefix='h100-final-publication-') as temporary:
        prepared, report = Path(temporary) / NAME, Path(temporary) / 'final-audit.md'
        try:
            build(args.source, args.nonuniform, args.attention, args.attention_source, prepared, report)
            from verify_final_audit import verify
            errors = verify(ROOT, prepared, report)
            if errors:
                raise ValueError('; '.join(errors))
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
        shutil.copytree(prepared, destination)
        shutil.copyfile(report, summary)
    print(f'Published separate audit: {summary}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
