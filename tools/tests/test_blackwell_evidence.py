"""A successful exit/throughput line cannot replace verified attention output."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_blackwell_appendix import BASE,COMPAT,attention_verified

class BlackwellEvidenceTests(unittest.TestCase):
    def setUp(self):self.text=(BASE/'results/forward-book-full-d64.log').read_text()
    def test_recorded_verified_output(self):self.assertTrue(attention_verified(0,self.text,64,2,'None'))
    def test_zero_exit_hopper_guard_is_not_a_kernel_test(self):
        guard=(COMPAT/'results/88_hopper_fmha/run-0.log').read_text()
        self.assertFalse(attention_verified(0,guard,64,2,'None'))
    def test_unverified_and_nonfinite_rates_rejected(self):
        self.assertFalse(attention_verified(0,self.text.replace('[OK]','[--]'),64,2,'None'))
        import re
        self.assertFalse(attention_verified(0,re.sub(r': [0-9.e+-]+ TFLOPS',': nan TFLOPS',self.text),64,2,'None'))
    def test_wrong_case_or_reference_failure_rejected(self):
        self.assertFalse(attention_verified(0,self.text,128,2,'None'))
        self.assertFalse(attention_verified(0,self.text+'\nfailed LSE: max diff 1\n',64,2,'None'))

if __name__=='__main__':unittest.main()
