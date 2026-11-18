#!/usr/bin/env python3
"""Verify the full attention listing, download and complete local-header bundle."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'docs/examples/attention/88_hopper_fmha'
PIN = '1afc6d355b8cb992398f17beb4425702f1d57ddd'
RUNNER_HASH = 'b29f795d77867fbac7954c55c39042d7a5b145659f84742c568c305cfc47b4c5'


def main():
    manifest = json.loads((FOLDER/'source-manifest.json').read_text())
    assert manifest['cutlass_commit'] == PIN
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((FOLDER/name).read_bytes()).hexdigest() == expected, name
    runner = (FOLDER/'88_hopper_fmha.cu').read_text()
    assert hashlib.sha256(runner.encode()).hexdigest() == RUNNER_HASH
    chapter = (ROOT/'docs/31-cutlass-hopper-fmha.md').read_text()
    shown = re.search(r'<!-- attention-complete-source:start -->\n```cuda\n(.*?)```\n<!-- attention-complete-source:end -->', chapter, re.S)
    presented = ''.join(line.rstrip() + '\n' for line in runner.splitlines())
    assert shown and shown[1] == presented, 'Full visible source differs from download beyond trailing whitespace'
    assert 'int main(' in runner
    with zipfile.ZipFile(FOLDER.parent/'hopper-fmha-source.zip') as archive:
        expected = {}
        for p in FOLDER.rglob('*'):
            if p.is_file():
                name = p.relative_to(FOLDER).as_posix()
                expected['88_hopper_fmha/'+name] = p.read_bytes()
        assert set(archive.namelist()) == set(expected)
        for name, data in expected.items():
            assert archive.read(name) == data, name
    print(f'PASS: complete 1192-line CUDA listing, pinned source identity, {len(manifest["files"])} source/license files and exact ZIP members; no CUDA execution.')


if __name__ == '__main__':
    main()
