#!/usr/bin/env python3
"""Check literal reader-comment corrections and their rendered results."""
import argparse
import hashlib
from html.parser import HTMLParser
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from verify_eight_gpu_audit import current_source


def verify(root, site=None):
    spec = importlib.util.spec_from_file_location('reader_comments', root / 'hooks/reader_comments.py')
    hook = importlib.util.module_from_spec(spec); spec.loader.exec_module(hook)
    source = current_source(root)
    before = hashlib.sha256(source.encode()).hexdigest()
    shown = hook.display_source(source)
    assert before == hashlib.sha256(current_source(root).encode()).hexdigest(), 'Stored source changed.'
    assert shown.count('\n') == source.count('\n'), 'Line numbers changed.'
    assert source.replace(hook.OLD, '', 1) == shown.replace(hook.NEW, '', 1), 'Non-comment material changed.'
    assert 'CHECK_CUDA(cudaGetDeviceCount(&n_gpus));' in shown, 'Checked query changed.'
    try:
        hook.display_source(source.replace(hook.OLD, hook.OLD.replace('Check GPU count', 'Different comment')))
    except ValueError:
        pass
    else:
        raise AssertionError('Nonmatching comment must fail closed.')
    page = SimpleNamespace(file=SimpleNamespace(src_path='another-page.md'))
    assert hook.on_page_markdown(source, page) == source, 'Unregistered page changed.'
    cluster = (root/'docs'/hook.CLUSTER_PAGE).read_text()
    shown_cluster = hook.on_page_markdown(cluster, SimpleNamespace(file=SimpleNamespace(src_path=hook.CLUSTER_PAGE)))
    assert cluster.count(hook.CLUSTER_OLD) == shown_cluster.count(hook.CLUSTER_NEW) == 1
    assert cluster.replace(hook.CLUSTER_OLD, '', 1) == shown_cluster.replace(hook.CLUSTER_NEW, '', 1)
    assert cluster.count('\n') == shown_cluster.count('\n')
    if site:
        class Text(HTMLParser):
            def __init__(self): super().__init__(); self.parts=[]
            def handle_data(self, data): self.parts.append(data)
        parser = Text(); parser.feed((site / '11-multi-gpu-nccl/index.html').read_text())
        text = ' '.join(' '.join(parser.parts).split())
        assert 'Check the visible device count before creating the eight-GPU layout.' in text, 'Reader comment absent.'
        assert "return code isn't checked here" not in text, 'Obsolete comment remains visible.'
        assert 'with zero CUDA-capable devices present.' not in text, 'Obsolete environment comment remains visible.'
        cluster_parser = Text(); cluster_parser.feed((site/'02-clusters-inline-ptx/index.html').read_text())
        cluster_text = ' '.join(' '.join(cluster_parser.parts).split())
        assert hook.CLUSTER_NEW in cluster_text and hook.CLUSTER_OLD not in cluster_text
    return before


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--site-dir', type=Path)
    a=p.parse_args(); value=verify(a.root.resolve(), a.site_dir.resolve() if a.site_dir else None)
    print('PASS: literal comment-only reader correction; executable source unchanged, SHA-256 ' + value)

if __name__ == '__main__': main()
