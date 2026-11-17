#!/usr/bin/env python3
"""Extract complete book programs and retain dated compiler/runtime evidence."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
OUT.mkdir(parents=True, exist_ok=True)
results = json.loads((OUT / 'results.json').read_text()) if (OUT / 'results.json').exists() else []
for page in sorted((ROOT / 'docs').glob('*.md')):
    if page.name.startswith('.'):
        continue
    source = page.read_text()
    for match in re.finditer(r'^```([^\n]*)\n(.*?)^```', source, re.M | re.S):
        code = match.group(2)
        if not (code.lstrip().startswith('//') and re.search(r'\bint\s+main\s*\(', code)):
            continue
        names = re.findall(r'\bex\w+\.cu\b', code)
        if not names:
            continue
        if len(sys.argv) > 1 and not any(value in page.name or value in names[0] for value in sys.argv[1:]):
            continue
        folder = OUT / page.stem / Path(names[0]).stem
        folder.mkdir(parents=True, exist_ok=True)
        cu = folder / names[0]
        cu.write_text(code)
        binary = folder / 'example'
        command = ['nvcc', '-std=c++17', '-arch=sm_90a', '-O2', str(cu), '-o', str(binary)]
        if 'cudaLaunchDevice' in code or 'cudaGridDependencySynchronize' in code:
            command += ['-rdc=true', '-lcudadevrt']
        if 'cuTensorMapEncode' in code:
            command += ['-lcuda']
        if '#include <nccl.h>' in code:
            command += ['-lnccl']
        if 'cublas' in code:
            command += ['-lcublas', '-Xcompiler', '-pthread']
        result = dict(page=str(page.relative_to(ROOT)), line=source[:match.start()].count('\n') + 1,
                      example=names[0], sha256=hashlib.sha256(code.encode()).hexdigest(),
                      timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), command=command)
        for stage, cmd, limit in [('compile', command, 120), ('run', [str(binary)], 90)]:
            if stage == 'run' and result.get('compile_exit') != 0:
                break
            try:
                process = subprocess.run(cmd, capture_output=True, text=True, timeout=limit)
                (folder / f'{stage}.log').write_text(process.stdout + process.stderr)
                result[f'{stage}_exit'] = process.returncode
            except subprocess.TimeoutExpired as exc:
                output = (exc.stdout or b'') + (exc.stderr or b'')
                (folder / f'{stage}.log').write_bytes(output)
                result[f'{stage}_exit'] = 'timeout'
        results = [old for old in results if (old['page'], old['example']) != (result['page'], result['example'])]
        results.append(result)
        (OUT / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(page.stem, names[0], result.get('compile_exit'), result.get('run_exit'), flush=True)
