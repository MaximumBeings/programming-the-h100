#!/usr/bin/env python3
"""Check the dated CPU attention record; does not compile CUTLASS or execute CUDA."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'docs/validation/2026-11-18-attention-cpu'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    inventory = json.loads((RECORD / 'integrity.json').read_text())
    expected = {'attention_walkthrough.py', 'commands.json', 'compile.log', 'run.log', 'system.json'}
    require(set(inventory) == expected, 'Unexpected CPU record inventory')
    for name, sha256 in inventory.items():
        require(digest(RECORD / name) == sha256, 'CPU record hash mismatch: ' + name)
    source = ROOT / 'docs/examples/advanced/attention_walkthrough.py'
    require(digest(source) == inventory['attention_walkthrough.py'], 'Current CPU source differs from captured revision')
    system = json.loads((RECORD / 'system.json').read_text())
    require(system['source_sha256'] == digest(source), 'System record identifies another source')
    require('no CUTLASS compilation or GPU execution' in system['scope'], 'Missing CPU-only scope')
    require(system['source'] == 'attention_walkthrough.py', 'Unexpected captured source name')
    for key in ['python', 'platform', 'machine']:
        require(isinstance(system.get(key), str) and bool(system[key].strip()), 'Missing system identity: ' + key)
    commands = json.loads((RECORD / 'commands.json').read_text())
    require([c['kind'] for c in commands] == ['compile', 'run'], 'Expected byte-compilation and CPU execution')
    chapter = (ROOT / 'docs/30-cutlass-attention-foundations.md').read_text()
    previous_finish = None
    recorded_interpreter = None
    recorded_directory = None
    for command in commands:
        kind = command['kind']
        expected_args = (['-m', 'py_compile', 'attention_walkthrough.py'] if kind == 'compile'
                         else ['attention_walkthrough.py'])
        argv = command['command']
        require(isinstance(argv, list) and len(argv) == len(expected_args) + 1
                and isinstance(argv[0], str)
                and re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?t?', Path(argv[0]).name)
                and argv[1:] == expected_args, 'Command does not execute the recorded CPU stage: ' + kind)
        require(command['log'] == kind + '.log', 'Stage uses the wrong captured log: ' + kind)
        directory = command['working_directory']
        require(isinstance(directory, str) and Path(directory).name == RECORD.name,
                'Command working directory does not identify this captured record')
        if recorded_interpreter is None:
            recorded_interpreter, recorded_directory = argv[0], directory
        require(argv[0] == recorded_interpreter and directory == recorded_directory,
                'Compile and run must use the same recorded interpreter and directory')
        require(command['exit_code'] == 0, 'Captured command failed')
        require(command['log_sha256'] == inventory[command['log']], 'Command log hash mismatch')
        times = [dt.datetime.fromisoformat(command[key]) for key in ['started_utc', 'finished_utc']]
        require(all(value.utcoffset() == dt.timedelta(0) for value in times), 'Timestamp is not UTC')
        require(times[0] <= times[1], 'Command finishes before it starts')
        require(previous_finish is None or previous_finish <= times[0], 'CPU execution precedes byte-compilation completion')
        previous_finish = times[1]
    output = (RECORD / 'run.log').read_text().rstrip()
    require('ATTENTION WALKTHROUGH PASS: CPU binary64 mathematics' in output, 'Missing checked CPU result')
    require(output in chapter, 'Chapter omits or changes captured CPU output')
    require(inventory['attention_walkthrough.py'] in chapter, 'Chapter omits exact CPU source identity')
    require(commands[1]['started_utc'] in chapter, 'Chapter omits CPU execution date')
    print('PASS: exact CPU attention source, dated commands, system information, immutable logs and displayed output; no GPU execution.')


if __name__ == '__main__':
    try:
        verify()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit('FAIL: ' + str(error))
