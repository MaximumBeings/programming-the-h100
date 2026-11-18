"""Host-only tests: these fixtures do not execute a CUDA kernel."""
import importlib.util
from pathlib import Path
import unittest
import os
import sys
import tempfile
import time

spec = importlib.util.spec_from_file_location('attention_collector', Path(__file__).parents[1] / 'run_cutlass_attention.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

HEADER = '###### B 1 H 2 Q 128 K 128 D 64 Forward Full #SM 132\n'
ROWS = (' [OK]  tma 64x128x64                  : 1.23 TFLOPS/s\n'
        ' [OK]  tma ws cooperative 128x64x64  : 1.24 TFLOPS/s\n'
        ' [OK]  tma ws ping-pong 128x64x64    : 1.25 TFLOPS/s\n')

class ClassifierTests(unittest.TestCase):
    def test_all_instantiated_variants_verified(self):
        self.assertEqual(module.classify(0, HEADER + ROWS, 3), 'PASS')

    def test_zero_exit_does_not_hide_failure(self):
        bad = HEADER + ROWS.replace(' [OK] ', '[FAIL] ', 1)
        self.assertEqual(module.classify(0, bad, 3), 'FAIL')
        self.assertEqual(module.classify(0, HEADER + ROWS + 'failed LSE: max diff 2 mean 1\nReference check failed', 3), 'FAIL')
        self.assertEqual(module.classify(0, HEADER + ROWS + 'failed dQ: max diff 2 mean 1', 3), 'FAIL')

    def test_partial_support_is_not_whole_case_pass(self):
        text = HEADER + ROWS[:ROWS.index(' [OK] ', 1)] + 'This kernel is not supported. Last CUDA error is: no error\n[FAIL] other : 0 TFLOPS/s\n'
        self.assertEqual(module.classify(0, text, 3), 'SKIP')

    def test_architecture_skip_and_macro_noop(self):
        self.assertEqual(module.classify(0, "This example requires a GPU of NVIDIA's Hopper Architecture (compute capability 90) and CUDA 12.0 or greater.\n", 3), 'SKIP')
        self.assertEqual(module.classify(0, HEADER, 3), 'INCOMPLETE')
        self.assertEqual(module.classify(0, '', 3), 'INCOMPLETE')

    def test_no_verify_is_not_verified(self):
        self.assertEqual(module.classify(0, HEADER + ROWS.replace('[OK]', '[--]'), 3), 'INCOMPLETE')
        self.assertEqual(module.classify(0, ROWS, 3), 'INCOMPLETE')
        self.assertEqual(module.classify(0, HEADER + ROWS, 2), 'INCOMPLETE')

    def test_backward_and_process_error(self):
        self.assertEqual(module.classify(0, HEADER.replace('Forward', 'Backward') + ROWS, 3), 'PASS')
        self.assertEqual(module.classify(1, HEADER + ROWS, 3), 'FAIL')

    def test_runtime_rows_must_match_requested_case(self):
        _, _, cases = module.commands(Path('/tmp/example-attention'), 1, 3, True)
        command = next(cmd for name, cmd, _ in cases if name == 'd64-full')
        rows = ''.join(line + '       t=0.1ms, smem=16384b\n' for line in ROWS.splitlines(keepends=True))
        text = HEADER + rows
        self.assertEqual(module.classify(0, text, 3, command), 'PASS')
        for wrong in [text.replace('H 2', 'H 4'), text.replace('Forward', 'Backward'),
                      text.replace('Full', 'Causal'), text.replace('128x64x64', '64x128x64'),
                      text.replace('t=0.1ms', 't=nanms', 1),
                      text.replace('1.23 TFLOPS/s', 'inf TFLOPS/s'),
                      text + HEADER]:
            self.assertEqual(module.classify(0, wrong, 3, command), 'INCOMPLETE', wrong)
        self.assertEqual(module.classify(0, text, 3, [a for a in command if a != '--verify']), 'INCOMPLETE')

    @unittest.skipUnless(os.name == 'posix', 'process-group check requires POSIX')
    def test_timeout_stops_descendant_and_retains_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            marker = directory / 'descendant-survived'
            child = "import time;from pathlib import Path;time.sleep(.6);Path(" + repr(str(marker)) + ").write_text('bad')"
            parent = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c'," + repr(child) + "]);print('started',flush=True);time.sleep(10)"
            result, text = module.record([sys.executable, '-c', parent], directory, 'timeout', .15)
            self.assertEqual(result['exit_code'], 124)
            self.assertIn('started', text)
            self.assertIn('process group terminated', text)
            time.sleep(.7)
            self.assertFalse(marker.exists())
            self.assertTrue((directory / 'timeout.stdout.log').is_file())
            self.assertTrue((directory / 'timeout.stderr.log').is_file())

    def test_planned_commands_are_explicit(self):
        _, _, cases = module.commands(Path('/tmp/example-attention'), 1, 3, True)
        self.assertEqual(len(cases), 8)
        for _, command, _ in cases:
            for key in ['--b=', '--h=', '--q=', '--k=', '--d=', '--mask=', '--iterations=']:
                self.assertTrue(any(arg.startswith(key) for arg in command), (key, command))
            self.assertIn('--verify', command)
        self.assertEqual(sum('--bwd' in command for _, command, _ in cases), 2)

if __name__ == '__main__':
    unittest.main()
