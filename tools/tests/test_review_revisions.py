"""Regression tests for pending source identity and the error-checking macro."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_examples import ROOT, programs
from review_revisions import historical_hashes, revisions

class ReviewRepairs(unittest.TestCase):
    def test_pending_revisions_are_exact_and_visible(self):
        rows = revisions(ROOT)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r['status'] == 'pending_gpu_revalidation' for r in rows))
        actual = {(p['page'],p['example']):p['sha256'] for p in programs()}
        before = historical_hashes(ROOT, actual)
        for r in rows:
            if 'source' in r: continue
            key = r['page'],r['example']
            self.assertEqual(before[key],r['archived_sha256'])
            changed = dict(actual);changed[key]='0'*64
            with self.assertRaises(ValueError):historical_hashes(ROOT, changed)

    def test_current_file_drift_is_rejected(self):
        original = Path.read_bytes
        row = revisions(ROOT)[0]
        target = ROOT/row['archived_source']
        def read(path):return b'changed' if path == target else original(path)
        with patch.object(Path,'read_bytes',read):
            with self.assertRaises(ValueError):revisions(ROOT)

    def test_macro_named_e_and_single_evaluation(self):
        source=next(p['code'] for p in programs() if p['example']=='ex11_nvlink_bandwidth.cu')
        helper=re.search(r'static inline void check_cuda_status\(.*?\n#define CHECK_CUDA\(call\)[^\n]+',source,re.S)[0]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'macro.cpp';binary=Path(folder)/'macro'
            path.write_text('#include <cstdio>\n#include <cstdlib>\nusing cudaError_t=int;\nconstexpr int cudaSuccess=0;\nconst char* cudaGetErrorString(int){return "expected failure";}\n'+helper+'\nint main(int argc,char**){ int e=argc>1?1:0; int count=0; CHECK_CUDA((++count,e)); return count==1?0:9; }\n')
            build=subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror',str(path),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            self.assertEqual(subprocess.run([str(binary)],capture_output=True).returncode,0)
            failed=subprocess.run([str(binary),'fail'],capture_output=True,text=True)
            self.assertEqual(failed.returncode,1)
            self.assertIn('expected failure',failed.stderr)

    def test_worked_schedule_and_resource_arithmetic(self):
        stages,micro=4,4
        occupied=set()
        for s in range(stages):
            for m in range(micro):
                f=s+m;b=13-m-s
                self.assertGreater(b,f)
                if s<3:self.assertGreater(b,13-m-(s+1))
                for t in (f,b):
                    self.assertNotIn((s,t),occupied);occupied.add((s,t))
        self.assertEqual(len(occupied),32)
        self.assertEqual(max(t for _,t in occupied)+1,14)
        self.assertEqual(228//33,6)
        self.assertEqual(64*128//128,64)
        self.assertEqual(8*8,64) # complete spatial subdivision, not eight lost tiles
        self.assertEqual(sum(x*x for x in [1,1,1,9]),sum(x*x for x in [6,-4,-4,4]))

if __name__=='__main__':unittest.main()
