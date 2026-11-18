#!/usr/bin/env python3
"""Verify the exact workshop sources, actual captures and rendered chapter records."""
import csv
import json
from pathlib import Path
import re
from publish_workshop import ROOT, DATASET, PAGES, sha, validate, section, supplemental, stage_ok

def verify_teaching_evidence(public, root):
    """Check the separate host probes and profiler evidence used by walkthroughs."""
    pin='1afc6d355b8cb992398f17beb4425702f1d57ddd'
    for directory,marker in [('layout-probe','COMPLEMENT PASS: 16 unique offsets'),
                             ('swizzle-probe','good(3,5)=102 bad(3,20)=212 errors=0')]:
        folder=public/directory
        record=json.loads((folder/'results.json').read_text())
        if (('cutlass_commit' in record and record['cutlass_commit']!=pin)
                or sha(folder/'probe.cu')!=record['source_sha256']):
            raise ValueError('Teaching probe source/pin differs: '+directory)
        for name in ('compile','run'):
            stage_ok(folder,dict(record[name],log=name+'.log'))
        if marker not in (folder/'run.log').read_text():
            raise ValueError('Teaching probe result missing: '+directory)
    folder=public/'optional-checks/profiler-checks/20261118T143847Z'
    records=json.loads((folder/'commands.json').read_text())
    if len(records)!=1 or records[0]['cutlass_commit']!=pin:
        raise ValueError('Profiler capture identity differs')
    record=records[0];stage_ok(folder,record)
    required={'--verification-enabled=true','--verification-required=true',
              '--verification-providers=reference_device','--beta=0.5'}
    if not required.issubset(record['command']):raise ValueError('Strict profiler verification missing')
    rows=list(csv.DictReader((folder/'cutlass-hopper.gemm.csv').open()))
    if (len(rows)!=1 or rows[0]['Disposition']!='passed' or rows[0]['Status']!='success'
            or 'reference_device: Passed' not in (folder/record['log']).read_text()
            or '--kernels='+rows[0]['Operation'] not in record['command']):
        raise ValueError('Actual profiler row/provider did not pass')
    if float(rows[0]['Runtime'])<=0:raise ValueError('Profiler runtime missing')
    analysis=public/'timeline-analysis'
    result=json.loads((analysis/'results.json').read_text())
    capture=json.loads((public/'capture-integrity.json').read_text())
    if (sha(analysis/'analyze_workshop_timeline.py')!=result['analysis_source_sha256']
            or sha(root/'tools/analyze_workshop_timeline.py')!=result['analysis_source_sha256']):
        raise ValueError('Timeline analysis source differs')
    for mode in ('serial','overlap'):
        row=result[mode]
        if (row['sqlite_sha256']!=capture[row['sqlite']] or row['kernels']!=128
                or row['copies']!=256 or row['payload_bytes']!=268435456):
            raise ValueError('Timeline counts/input capture identity differ')
    if (result['serial']['kernels_intersecting_other_stream_copy']!=0
            or result['overlap']['kernels_intersecting_other_stream_copy']!=97):
        raise ValueError('Published cross-stream overlap counts differ')


def verify_stride_record(root):
    folder=root/'docs/validation/2026-11-18-strides'
    inventory=json.loads((folder/'integrity.json').read_text())
    files={p.name for p in folder.iterdir() if p.is_file()}-{'integrity.json'}
    if files!=set(inventory):raise ValueError('Stride inventory differs')
    for name,digest in inventory.items():
        if Path(name).name!=name or sha(folder/name)!=digest:raise ValueError('Stride evidence changed')
    source=root/'docs/examples/advanced/stride_walkthrough.py'
    if sha(source)!=inventory['stride_walkthrough.py']:raise ValueError('Current stride source differs')
    records=json.loads((folder/'commands.json').read_text())
    if len(records)!=2 or {r['kind'] for r in records}!={'compile','run'}:raise ValueError('Stride stages missing')
    for record in records:
        if record['exit_code']!=0 or sha(folder/record['log'])!=record['log_sha256']:
            raise ValueError('Stride command failed or log changed')
    text=(root/'docs/16-cute-strides.md').read_text()
    output=(folder/'run.log').read_text().strip()
    if 'STRIDE WALKTHROUGH PASS' not in output or output not in text or inventory['stride_walkthrough.py'] not in text:
        raise ValueError('Stride chapter output/source identity differs')


def verify(root=ROOT):
    verify_stride_record(root)
    public=root/'docs/validation'/DATASET
    inventory=json.loads((public/'integrity.json').read_text())
    actual={p.relative_to(public).as_posix() for p in public.rglob('*') if p.is_file()}-{'integrity.json'}
    if inventory.get('dataset')!=DATASET or actual!=set(inventory['files']):raise ValueError('Workshop inventory differs')
    for name,digest in inventory['files'].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or sha(public/rel)!=digest:raise ValueError('Changed/unsafe workshop evidence: '+name)
    summary=root/'docs/cutlass-workshop-validation.md'
    if sha(summary)!=inventory['summary_sha256']:raise ValueError('Workshop summary hash differs')
    rows,system,profiles=validate(public,root)
    verify_teaching_evidence(public,root)
    extra=supplemental(public,root)
    if extra and extra not in summary.read_text():raise ValueError('Supplemental claims differ from evidence')
    for row in rows:
        stem=Path(row['source']).stem;text=(root/'docs'/PAGES[stem]).read_text()
        parts=re.findall(r'<!-- workshop:start:'+re.escape(stem)+r' -->(.*?)<!-- workshop:end:'+re.escape(stem)+r' -->',text,re.S)
        if len(parts)!=1 or ('<!-- workshop:start:'+stem+' -->'+parts[0]+'<!-- workshop:end:'+stem+' -->')!=section(row,public,root):raise ValueError('Workshop chapter source/commands/dates/output differ: '+stem)
    return rows
if __name__=='__main__':
    rows=verify()
    print('PASS: eight CUDA translation units, two Python references, eleven successful invocations, two real Systems captures, exact sources/dates/output and immutable inventory.')
