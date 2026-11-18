"""Reject silent source drift in explicitly pending GPU corrections."""
import json
import unittest
from pathlib import Path
from pending_revalidation import pending
from validate_examples import programs

ROOT = Path(__file__).resolve().parents[1]

class PendingTests(unittest.TestCase):
    def test_pinned_current_and_historical_revisions(self):
        current = {(p['page'], p['example']): p['sha256'] for p in programs()}
        rows = json.loads((ROOT/'docs/validation/2026-11-17-final-audit/results.json').read_text())
        recorded = {(p['page'], p['example']): p['sha256'] for p in rows}
        revisions = pending(ROOT, current, recorded)
        self.assertEqual(len(revisions), 4)
        key = next(iter(revisions))
        for bad in ('0'*64, recorded[key]):
            changed = dict(current); changed[key] = bad
            with self.assertRaises(ValueError): pending(ROOT, changed, recorded)
        changed = dict(recorded); changed[key] = "f"*64
        with self.assertRaises(ValueError): pending(ROOT, current, changed)

if __name__ == '__main__': unittest.main()
