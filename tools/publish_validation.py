#!/usr/bin/env python3
"""Publish captured evidence beside the corresponding book examples."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / 'validation' / '2026-11-17'
PUBLIC = ROOT / 'docs' / 'validation' / '2026-11-17'
PUBLIC.mkdir(parents=True, exist_ok=True)
rows = json.loads((RECORDS / 'results.json').read_text())
for page in sorted((ROOT / 'docs').glob('*.md')):
    text = page.read_text()
    programs = [row for row in rows if row['page'] == str(page.relative_to(ROOT))]
    for row in sorted(programs, key=lambda item: item['line'], reverse=True):
        name = row['example']
        source_folder = RECORDS / page.stem / Path(name).stem
        destination = PUBLIC / page.stem / Path(name).stem
        destination.mkdir(parents=True, exist_ok=True)
        for filename in (name, 'compile.log', 'run.log'):
            shutil.copy2(source_folder / filename, destination / filename)
        matches = list(re.finditer(r'^```[^\n]*\n(.*?)^```', text, re.M | re.S))
        code = next(match for match in matches if match[1].lstrip().startswith('//')
                    and name in match[1] and re.search(r'\bint\s+main\s*\(', match[1]))
        if hashlib.sha256(code[1].encode()).hexdigest() != row['sha256']:
            raise ValueError(f'Unvalidated source change: {name}')
        skipped = page.name.startswith('11-')
        status = 'Compile PASS; multi-GPU workload NOT RUN (one GPU available)' if skipped else 'Compile PASS; execution PASS'
        if row['compile_exit'] != 0 or (not skipped and row.get('run_exit') != 0):
            raise ValueError(f'Cannot publish a passing result: {name}')
        command = row['command'][:]
        command = [name if arg.endswith('/' + name) else Path(name).stem if arg.endswith('/example') else arg for arg in command]
        url = f'validation/2026-11-17/{page.stem}/{Path(name).stem}'
        record = f'''<!-- validation:start:{name} -->
**Captured output — {name}**

> **Executed:** {row['timestamp']} (UTC). **Status:** {status}.
> **System:** 1× NVIDIA H100 PCIe, 114 SMs, 80 GB class; Ubuntu 24.04.5 LTS, NVIDIA driver 580.105.08, CUDA Toolkit 12.8.93, C++17, `sm_90a`.
> Compiler exit: `{row['compile_exit']}`; process exit: `{row.get('run_exit')}`. Timings and print ordering are specific to this run.

Save the [tested source]({url}/{name}) as `{name}`, then run:

```bash
{shlex.join(command)}
./{Path(name).stem}
```

[Compiler log]({url}/compile.log) · [Execution log]({url}/run.log)

```text
{chr(10).join(line.rstrip() for line in (source_folder / 'run.log').read_text().splitlines())}
```
<!-- validation:end:{name} -->
'''
        tail = text[code.end():]
        existing = re.search(r'<!-- validation:start:' + re.escape(name) + r' -->.*?<!-- validation:end:' + re.escape(name) + r' -->', tail, re.S)
        if existing:
            text = text[:code.end() + existing.start()] + record + text[code.end() + existing.end():]
            continue
        sample = re.search(r'\*\*Sample [Oo]utput[^\n]*\n', tail)
        if not sample:
            raise ValueError(f'Missing sample output position: {name}')
        output = re.search(r'^```[^\n]*\n.*?^```', tail[sample.end():], re.M | re.S)
        if not output:
            raise ValueError(f'Missing sample output fence: {name}')
        begin = code.end() + sample.start()
        end = code.end() + sample.end() + output.end()
        text = text[:begin] + record + text[end:]
    if programs:
        text = re.sub(r'^> \*\*Verified on:\*\*[^\n]*\n', '', text, flags=re.M)
    page.write_text(text)
shutil.copy2(RECORDS / 'results.json', PUBLIC / 'results.json')
print(f'Published {len(rows)} dated compilation/execution records.')
