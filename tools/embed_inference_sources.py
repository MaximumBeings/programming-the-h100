#!/usr/bin/env python3
"""Keep complete, copyable chapter listings identical to runnable downloads."""
import argparse
import ast
import re
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/examples/inference-systems'
BEGIN = '<!-- BEGIN COMPLETE INFERENCE SOURCE -->'
END = '<!-- END COMPLETE INFERENCE SOURCE -->'
TRACES = {
    32: ('Req and CostModel', 'simulate', 'summarize and run_tests'),
    33: ('skewed_router and expert_loads', 'eplb_place and check_alltoall', 'run_tests'),
    34: ('restore_seconds and recompute_seconds', 'TierCache', 'run_tests and extras'),
    35: ('transfer_ms and decode_config', 'sim_colocated and sim_disagg_pool', 'run_tests'),
    36: ('Rec and Target', 'one_request and run_open_loop', 'summarize and selftests'),
    37: ('Model and step_time', 'kv_budget_tokens and speedup', 'selftests and tables'),
    38: ('World and SemanticCache', 'run_stream and calibrate', 'selftests and calib_selftest'),
    39: ('router_feature and evaluate', 'calibrate_tau', 'selftests and experiments'),
    40: ('Scaler', 'simulate and summarize', 'selftests and experiments'),
    41: ('CircuitBreaker and RetryBudget', 'run_client and simulate_fleet', 'selftests'),
    42: ('Cfg', 'gateway and summarize_cfg', 'selftests and experiments'),
    43: ('BlockAllocator and PagedKVCache', 'dense_attention and paged_attention', 'selftests'),
    44: ('Req', 'simulate_continuous and simulate_static', 'selftests and summarize'),
    45: ('Node and Handle', 'RadixCache and BlockHashCache', 'selftests'),
    46: ('TinyLM and KVPool', 'PrefixCache and Engine', 'reference and selftests'),
}


def pages():
    result = []
    for n in range(32, 48):
        page = next((ROOT/'docs').glob(f'{n}-*.md'))
        files = sorted(SOURCE.glob(f'ch{n}_*.py'))
        result.append((n, page, files))
    result.append(('FP4', ROOT/'docs/appendix-blackwell-fp4.md', [SOURCE/'fp4_formats.py']))
    return result


def dependencies(files):
    local = {p.stem:p for p in SOURCE.glob('*.py')}
    result = set()
    for file in files:
        for node in ast.walk(ast.parse(file.read_text())):
            names = [node.module] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names] if isinstance(node,ast.Import) else []
            for name in names:
                if name in local and local[name] not in files:
                    result.add(local[name])
    return sorted(result)


def section(n, files):
    deps = dependencies(files)
    lines = [BEGIN, '## Complete runnable code', '',
             'The complete implementation is shown below, including its checks and entry point. '
             'The listing and downloadable `.py` file contain exactly the same source. '
             'Copy the file under the displayed filename; keep supporting files in the same directory.', '',
             '[Download all inference examples and supporting modules](examples/inference-systems/inference-systems.zip) '
             'or read the [environment and command guide](examples/inference-systems/README.md).', '',
             'The [follow-up agent audit](inference-adversarial-review-2026-11-25.md) records the corrections, fresh CPU captures and hardware limits for this source.', '']
    if deps:
        lines += ['Supporting modules: '+', '.join(f'[{p.name}](examples/inference-systems/{p.name})' for p in deps)+'.', '']
    if n in TRACES:
        first,operation,check=TRACES[n]
        fmt=lambda value:' and '.join('`'+part+'`' for part in value.split(' and '))
        lines += [f'Read it in three passes: start with {fmt(first)} to identify the state; '
                  f'trace {fmt(operation)} for one request or token; then follow {fmt(check)} to see how the result is checked.', '']
    if n != 47:
        name=files[0].name
        opt='--test' if n in (32,33,34,35) else '--selftest' if n in (36,38,39,40) else 'tests' if n in (41,42,43,44,45,46) else ''
        commands=[f'python3 docs/examples/inference-systems/{name}'+(' '+opt if opt else '')]
        if opt:
            commands += ['# Run the full chapter demonstrations:',f'python3 docs/examples/inference-systems/{name}']
        lines += ['From the repository root:', '', '```bash',*commands,'```','']
    else:
        lines += ['The CUDA program requires a CUDA GPU. Start the Chapter 47 vLLM server before running the HTTP probe; '
                  'its `--model` and `--output` arguments are required.', '', '```bash',
                  'python3 docs/examples/inference-systems/ch47_paged_attention_cuda.py',
                  'python3 docs/examples/inference-systems/ch47_vllm_probe.py \\',
                  '  --model Qwen/Qwen2.5-0.5B-Instruct --output probe.json', '```', '']
    for file in files:
        lines += [f'### {file.name}', '',f'[Download {file.name}](examples/inference-systems/{file.name})', '',
                  f'```python title="{file.name}" linenums="1"',file.read_text().rstrip(), '```', '']
    lines += [END, '']
    return '\n'.join(lines)


def expected_page(n, page, files):
    text=page.read_text()
    if BEGIN in text:
        return re.sub(re.escape(BEGIN)+r'.*?'+re.escape(END),lambda _:section(n,files).rstrip(),text,flags=re.S)
    # Replace the previous Chapter 47-only full listings with the uniform section.
    if n==47:
        text=re.sub(r'\n## Complete lab sources\n.*?(?=\n## Exercises with worked answers)', '\n',text,flags=re.S)
    marker=re.search(r'^## (?:\d+\.\d+\s+)?(?:Programming exercises|Exercises|Coding Exercises|Chapter Summary)\b',text,re.M|re.I)
    pos=marker.start() if marker else len(text)
    return text[:pos]+section(n,files)+'\n'+text[pos:]


def write_bundle(check=False):
    target=SOURCE/'inference-systems.zip'
    members={p.name:p.read_bytes() for p in SOURCE.glob('*.py')}
    members['README.md']=(SOURCE/'README.md').read_bytes()
    members['requirements-reference.txt']=(SOURCE/'requirements-reference.txt').read_bytes()
    if check:
        with zipfile.ZipFile(target) as z:
            assert set(z.namelist())==set(members),'bundle member drift'
            for name,data in members.items():assert z.read(name)==data,f'bundle source drift: {name}'
    else:
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for name,data in sorted(members.items()):
                info=zipfile.ZipInfo(name,date_time=(2026,11,25,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
                z.writestr(info,data)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    for n,page,files in pages():
        result=expected_page(n,page,files)
        if args.check:assert page.read_text()==result,f'inline source drift: {page.name}'
        else:page.write_text(result)
    write_bundle(args.check)
    print('PASS: 18 complete source listings across Chapters 32–47 and FP4 match their downloads and bundled dependencies')

if __name__=='__main__':main()
