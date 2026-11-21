#!/usr/bin/env python3
"""Verify attention source/display and actual H100 records without running CUDA."""
from pathlib import Path
import hashlib
import json
import re
import run_cutlass_attention as collector

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'docs/validation/2026-11-18-h100-attention'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(folder):
    data = json.loads((folder/'integrity.json').read_text())
    require(set(data) == {p.name for p in folder.iterdir() if p.is_file() and p.name != 'integrity.json'}, 'Evidence inventory differs from files')
    for name, expected in data.items():
        require(Path(name).name == name, 'Unexpected evidence path')
        require(sha(folder/name) == expected, str(folder/name)+' hash mismatch')
    return data


def output(folder, command):
    text = ''
    require(command['exit_code'] == 0, 'Failed recorded command')
    for key in ['stdout', 'stderr']:
        path = folder/command[key]
        if key+'_sha256' in command:
            require(sha(path) == command[key+'_sha256'], 'Command log hash mismatch')
        text += path.read_text()+'\n'
    return text


def verify():
    chapter30 = (ROOT/'docs/30-cutlass-attention-foundations.md').read_text()
    chapter31 = (ROOT/'docs/31-cutlass-hopper-fmha.md').read_text()
    row = BASE/'online-row-01'
    inventory(row)
    result = json.loads((row/'results.json').read_text())
    require(result['status'] == 'PASS', 'CUDA row record did not pass')
    source = ROOT/'docs/examples/attention/attention_online.cu'
    from review_revisions import historical_file_hash
    require(historical_file_hash(ROOT, str(source.relative_to(ROOT))) == sha(row/'attention_online.cu'), 'CUDA historical identity differs from tested revision')
    marker = re.search(r'<!-- attention-row-source:start -->\n```cuda\n(.*?)```\n<!-- attention-row-source:end -->', chapter30, re.S)
    require(marker is not None and marker[1] == source.read_text(), 'Chapter omits/changes complete CUDA source')
    py = ROOT/'docs/examples/advanced/attention_walkthrough.py'
    marker = re.search(r'<!-- attention-python-source:start -->\n```python\n(.*?)```\n<!-- attention-python-source:end -->', chapter30, re.S)
    require(marker is not None and marker[1] == py.read_text(), 'Chapter omits/changes complete Python source')
    require(py.read_bytes() == (row/'attention_walkthrough.py').read_bytes(), 'Python source differs from H100 rerun')
    commands = {c['label']: c for c in result['commands']}
    for c in commands.values():output(row, c)
    require(commands['compile']['argv'][0] == 'nvcc' and '-arch=sm_90a' in commands['compile']['argv'], 'CUDA compilation command mismatch')
    captured = (row/commands['execute']['stdout']).read_text().rstrip()
    require(captured in chapter30 and 'CUDA ATTENTION PASS: seven cases' in captured, 'Displayed GPU row output mismatch')
    require(captured.count('PASS worked-') == 4 and captured.count('PASS rectangular-') == 3, 'Missing CUDA row cases')
    mem_output = output(row, commands['memcheck'])
    require(commands['memcheck']['argv'] == ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '3', './attention_online'], 'Row memcheck command changed')
    require('ERROR SUMMARY: 0 errors' in mem_output and 'CUDA ATTENTION PASS: seven cases' in mem_output, 'Row memory check not clean')
    require('ATTENTION WALKTHROUGH PASS' in output(row, commands['python-execute']), 'Python rerun not checked')
    require(historical_file_hash(ROOT, str(source.relative_to(ROOT))) in chapter30 and commands['execute']['started_utc'] in chapter30, 'Missing source identity/date')
    require(result['system']['CUDA_VISIBLE_DEVICES'] == '1', 'Row GPU selection changed')

    fmha = BASE/'hopper-fmha'
    inventory(fmha)
    result = json.loads((fmha/'results.json').read_text())
    require(result['status'] == 'PASS' and result['cutlass_commit'] == collector.PIN, 'FMHA status/revision mismatch')
    require(sha(fmha/'collector.py') == result['collector_sha256'], 'Recorded collector hash mismatch')
    require(result['system']['environment']['CUDA_VISIBLE_DEVICES'] == '0', 'FMHA GPU selection changed')
    source_hashes = json.loads((fmha/'source-sha256.json').read_text())
    require(len(source_hashes) == 6711, 'Pinned compiler-input inventory changed')
    manifest = json.loads((ROOT/'docs/examples/attention/88_hopper_fmha/source-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        upstream = manifest.get('upstream_names', {}).get(name, name)
        key = 'LICENSE.txt' if name == 'LICENSE.txt' else collector.EXAMPLE+'/'+upstream
        require(source_hashes.get(key) == expected, 'Published project differs from compiled source: '+name)
    expected_cases = collector.commands(Path('/not-executed'), 4, 3, True)[2]
    require(len(result['cases']) == 8, 'Expected six forward/two backward invocations')
    for case, (name, command, count) in zip(result['cases'], expected_cases):
        require(case['name'] == name and case['expected_verified_variants'] == count, 'FMHA case inventory mismatch')
        require(case['argv'][1:] == command[1:], 'FMHA runtime arguments changed')
        text = output(fmha, case)
        require(case['status'] == 'PASS' and collector.classify(0, text, count, case['argv']) == 'PASS', 'FMHA numerical verification failed')
        require(name in chapter31 and case['started_utc'] in chapter31, 'Chapter omits FMHA case/date')
    for c in result['records']:output(fmha, c)
    for name in ['d64-full', 'd64-backward-no']:
        require((fmha/(name+'.stdout.log')).read_text().rstrip() in chapter31, 'Captured FMHA sample omitted/changed')
    require(sum(c['expected_verified_variants'] for c in result['cases']) == 17, 'Expected 17 verified FMHA variants')

    memory = BASE/'attention-memory-01'
    inventory(memory)
    result = json.loads((memory/'results.json').read_text())
    require(result['status'] == 'PASS' and len(result['commands']) == 4, 'Supplemental check inventory mismatch')
    for c in result['commands']:
        text = output(memory, c)
        if c.get('status'):
            require(c['argv'][:5] == ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '3'], 'FMHA memcheck command changed')
            expected = 2 if '--bwd' in c['argv'] else 3
            require(c['status'] == 'PASS' and collector.classify(0, text, expected, c['argv'][5:]) == 'PASS' and 'ERROR SUMMARY: 0 errors' in text, 'FMHA memory check failed')
        else:
            device = json.loads((memory/c['stdout']).read_text())
            require(device['CUDA_VISIBLE_DEVICES'] == c['CUDA_VISIBLE_DEVICES'] and device['uuid'].startswith('GPU-'), 'Missing CUDA device identity')
    print('PASS: complete current listings and exact historical captures: seven H100 row cases, 17 FMHA variants and three memchecks. Corrected attention source remains GPU-pending.')


if __name__ == '__main__':
    try:
        verify()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit('FAIL: '+str(error))
