#!/usr/bin/env python3
"""Publish a PASS-only eight-A100 supplement; never alter existing evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
from verify_eight_gpu_audit import NAME, KEY, MARKER, check_data, current_source, sha, verify


def publish(root, evidence):
    errors = check_data(evidence)
    m = json.loads((evidence / 'manifest.json').read_text())
    if hashlib.sha256(current_source(root).encode()).hexdigest() != m['source_sha256']:
        errors.append('Current chapter source does not match executed source.')
    if errors:
        raise ValueError('; '.join(errors))
    public = root / 'docs/validation' / NAME
    summary = root / 'docs/eight-gpu-validation.md'
    if public.exists() or summary.exists():
        raise ValueError('Published supplement already exists; immutable evidence is never overwritten.')
    public.mkdir(parents=True)
    names = ['manifest.json', KEY[1], 'compile.log', 'run.log', 'system-info.log']
    for name in names:
        shutil.copyfile(evidence / name, public / name)
    command = m['compile']['command']
    normalized = [KEY[1] if str(x).endswith('/' + KEY[1]) else 'ex13_hybrid_parallel_training' if i and command[i-1] == '-o' else x for i,x in enumerate(command)]
    base = 'validation/' + NAME
    text = f'''# Eight-GPU A100 training validation

In the dated execution below, the portable Chapter 11 TP=2 × PP=2 × DP=2 teaching program completed on **eight NVIDIA A100 GPUs**, compiled for **`sm_80`**. This is a separate architecture-specific supplement; the original one-H100 and final two-H100 datasets remain unchanged. It is not an eight-H100 run. Hopper-specific WGMMA and TMA examples are not validated by this A100 execution.

- **Run captured:** `{m['timestamp_utc']}` (UTC).
- **Finished:** `{m['finished_utc']}` (UTC).
- **Compiler / process exit:** `0` / `0`.
- **Program result:** `{MARKER}`.
- **Executed source SHA-256:** `{m['source_sha256']}`.

The revised program uses distinct DP inputs and checks all 16,384 final weight values against an independent CPU serial reference, plus all 16 per-step replica loss values and four DP replica-pair comparisons. The reference follows the same DP averaging and per-TP-shard gradient-clipping semantics, with explicit numerical tolerances. This is a reference comparison of the teaching program, not a training-quality or throughput benchmark, cross-node RDMA test, or validation on B200. Loss is checked against the reference for each step; changing inputs do not imply a monotonic loss trajectory.

Download the [exact executed source]({base}/{KEY[1]}), [manifest]({base}/manifest.json), [compiler log]({base}/compile.log), [raw run log]({base}/run.log), and [system/topology log]({base}/system-info.log). The complete raw log is authoritative; thread print order is run-specific.

The captured compiler command, with only the remote source and output paths shortened, is:

```bash
{shlex.join(normalized)}
./ex13_hybrid_parallel_training
```

The manifest preserves the exact original absolute-path commands and UTC stage timestamps. Verify the immutable hashes and current chapter-source match with:

```bash
python3 tools/verify_eight_gpu_audit.py
```

The [complete validation summary](final-audit.md) combines the 26 successful H100 program runs with this eight-A100 training result: all 27 complete programs compiled and executed successfully across the recorded hosts.
'''
    summary.write_text(text)
    inventory = dict(files={name: sha(public / name) for name in names}, summary_sha256=sha(summary))
    (public / 'integrity.json').write_text(json.dumps(inventory, indent=2) + '\n')
    errors = verify(root)
    if errors:
        raise ValueError('Publication failed verification: ' + '; '.join(errors))
    print('PASS: published separate immutable eight-A100 supplement.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--evidence', type=Path, required=True)
    a = p.parse_args()
    publish(a.root.resolve(), a.evidence.resolve())

if __name__ == '__main__':
    main()
