#!/usr/bin/env python3
"""Check the dated CPU attention record; does not compile CUTLASS or execute CUDA."""
import datetime as dt
import hashlib
import json
from pathlib import Path

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
    commands = json.loads((RECORD / 'commands.json').read_text())
    require([c['kind'] for c in commands] == ['compile', 'run'], 'Expected byte-compilation and CPU execution')
    chapter = (ROOT / 'docs/30-cutlass-attention-foundations.md').read_text()
    for command in commands:
        require(command['exit_code'] == 0, 'Captured command failed')
        require(command['log_sha256'] == inventory[command['log']], 'Command log hash mismatch')
        for key in ['started_utc', 'finished_utc']:
            require(dt.datetime.fromisoformat(command[key]).utcoffset() == dt.timedelta(0), 'Timestamp is not UTC')
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
