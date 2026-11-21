"""Check November 21 reruns against current sources and immutable raw evidence."""
import hashlib,json
from pathlib import Path
from validate_examples import programs
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/validation/2026-11-21-two-h100'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verified_sources(root=ROOT):
 base=root/'docs/validation/2026-11-21-two-h100'
 inventory=json.loads((base/'checksums.json').read_text())
 for rel,digest in inventory.items():
  p=Path(rel)
  if p.is_absolute() or '..' in p.parts or sha(base/p)!=digest:raise ValueError('Evidence hash/path mismatch: '+rel)
 first=json.loads((base/'h100-rerun-20261121/results.json').read_text())
 second=json.loads((base/'h100-two-gpu-20261121/results.json').read_text())
 current={(r['page'],r['example']):r for r in programs()}
 if len(first)!=27 or len(second)!=2:raise ValueError('Changed rerun inventory')
 merged={}
 for folder,rows in [('h100-rerun-20261121',first),('h100-two-gpu-20261121',second)]:
  for row in rows:
   key=row['page'],row['example'];source=base/folder/Path(row['page']).stem/Path(row['example']).stem/row['example']
   if key not in current or row['sha256']!=current[key]['sha256'] or sha(source)!=row['sha256']:raise ValueError('Current source identity mismatch: '+str(key))
   if row['compile_exit']!=0 or '-arch=sm_90a' not in row['command']:raise ValueError('Compilation not successful')
   if row['status']=='PASS':
    if row['run_exit']!=0 or not row.get('run_timestamp'):raise ValueError('Run not successful')
    merged[key]=row['sha256']
 if len(merged)!=26 or set(current)-set(merged)!={('docs/11-multi-gpu-nccl.md','ex13_hybrid_parallel_training.cu')}:raise ValueError('Expected 26 executed sources and eight-GPU exclusion')
 sup=base/'h100-supplement-20261121';rows=json.loads((sup/'manifest.json').read_text());checks={r['name']:r for r in rows}
 wanted=[name+'-'+tool for name in ['ex01_cluster_inspect','ex02_ptx_basics','ex06_comprehensive','ex02_reduce_async'] for tool in ['memcheck','racecheck','synccheck']]+['attention-memcheck']
 for name in wanted:
  row=checks[name];text=(sup/row['log']).read_text()
  if row['exit_code']!=0 or sha(sup/row['log'])!=row['log_sha256']:raise ValueError('Sanitizer command failed: '+name)
  marker='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if name.endswith('racecheck') else 'ERROR SUMMARY: 0 errors'
  if marker not in text:raise ValueError('Sanitizer clean summary absent: '+name)
 source=base/'h100-attention-20261121/attention_online.cu'
 if sha(source)!=sha(root/'docs/examples/attention/attention_online.cu'):raise ValueError('Current attention identity differs')
 if 'CUDA ATTENTION PASS: seven cases' not in (sup/'attention-memcheck.log').read_text():raise ValueError('Missing attention numerical checks')
 for page in {key[0] for key in merged} | {'docs/30-cutlass-attention-foundations.md','docs/review-fixes-2026-11-21.md','docs/validation-b200.md'}:
  text=(root/page).read_text()
  for stale in ['GPU revalidation pending for the current source','GPU rerun pending.','current corrections and pending reruns']:
   if stale in text:raise ValueError('Stale pending claim after successful rerun: '+page)
 merged[('docs/30-cutlass-attention-foundations.md','attention_online.cu')]=sha(source)
 return merged
if __name__=='__main__':
 verified_sources();print('PASS: 26 current complete programs executed; seven-case attention and 13 clean sanitizer checks. Eight-GPU ex13 remains unexecuted.')
