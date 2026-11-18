"""Host-only semantic checks for the immutable CPU attention record verifier.

Every negative fixture updates its own commands checksum, so rejection must come
from the command/record contract rather than a trivial stale digest. Original
records and the mathematical oracle are never modified.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RECORD_NAME = '2026-11-18-attention-cpu'


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        'attention_record_fixture', ROOT / 'tools/verify_attention_walkthrough.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AttentionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='attention-evidence-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.record = self.root / 'docs/validation' / RECORD_NAME
        shutil.copytree(ROOT / 'docs/validation' / RECORD_NAME, self.record)
        source_directory = self.root / 'docs/examples/advanced'
        source_directory.mkdir(parents=True)
        shutil.copy2(ROOT / 'docs/examples/advanced/attention_walkthrough.py',
                     source_directory / 'attention_walkthrough.py')
        shutil.copy2(ROOT / 'docs/30-cutlass-attention-foundations.md',
                     self.root / 'docs/30-cutlass-attention-foundations.md')
        self.verifier = load_verifier()
        self.verifier.ROOT = self.root
        self.verifier.RECORD = self.record

    def verify(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.verifier.verify()

    def rewrite_commands(self, mutate):
        path = self.record / 'commands.json'
        commands = json.loads(path.read_text())
        mutate(commands)
        path.write_text(json.dumps(commands))
        inventory_path = self.record / 'integrity.json'
        inventory = json.loads(inventory_path.read_text())
        inventory['commands.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory_path.write_text(json.dumps(inventory))

    def reject(self, mutate, expected_message):
        self.rewrite_commands(mutate)
        with self.assertRaisesRegex(ValueError, expected_message):
            self.verify()

    def test_original_record_passes(self):
        self.verify()

    def test_wrong_program_rejected_after_rehash(self):
        self.reject(lambda c: c[1].update(command=['/usr/bin/echo', 'attention_walkthrough.py']),
                    'Command does not execute')

    def test_wrong_source_rejected_after_rehash(self):
        self.reject(lambda c: c[1].update(command=[c[1]['command'][0], 'another.py']),
                    'Command does not execute')

    def test_reversed_duration_rejected_after_rehash(self):
        self.reject(lambda c: c[1].update(finished_utc='2026-11-18T21:34:24+00:00'),
                    'finishes before it starts')

    def test_run_before_compile_rejected_after_rehash(self):
        self.reject(lambda c: c[1].update(started_utc='2026-11-18T21:34:24+00:00'),
                    'precedes byte-compilation')

    def test_swapped_log_rejected_after_rehash(self):
        self.reject(lambda c: c[1].update(log='compile.log',
                    log_sha256=hashlib.sha256(b'').hexdigest()), 'wrong captured log')

    def test_different_interpreter_rejected_after_rehash(self):
        self.reject(lambda c: c[1].update(command=['/usr/bin/python3', 'attention_walkthrough.py']),
                    'same recorded interpreter')

    def test_wrong_directory_rejected_after_rehash(self):
        self.reject(lambda c: c[1].update(working_directory='/tmp/unrelated'),
                    'working directory does not identify')


if __name__ == '__main__':
    unittest.main()
