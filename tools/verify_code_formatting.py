#!/usr/bin/env python3
"""Verify all formatted source listings and, optionally, their built HTML."""
import argparse
from html.parser import HTMLParser
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'hooks' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Listings(HTMLParser):
    def __init__(self):
        super().__init__()
        self.divs, self.parts, self.blocks = [], None, []
        self.pre = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'div':
            self.divs.append(attrs.get('class', '').split())
        elif tag == 'pre':
            self.pre = True
        elif tag == 'code' and self.pre:
            self.parts = []

    def handle_endtag(self, tag):
        if tag == 'code' and self.parts is not None:
            classes = {c for level in self.divs for c in level}
            if classes & {'language-c', 'language-cpp', 'language-cuda'}:
                self.blocks.append(''.join(self.parts))
            self.parts = None
        elif tag == 'pre':
            self.pre = False
        elif tag == 'div':
            self.divs.pop()

    def handle_data(self, data):
        if self.parts is not None:
            self.parts.append(data)


def verify(site=None):
    hook = load('code_formatting')
    # Guard against a whitespace-only comparison accidentally accepting changed
    # operators, literals or comments; keep escaped macro newlines supported.
    for before, after in [('a << b;', 'a < < b;'), ('"a b"', '"ab"'),
                          ('x++;', 'x + +;'), ('// first\nint x;', '// second\nint x;')]:
        assert hook.lexical_signature(before) != hook.lexical_signature(after)
    assert hook.lexical_signature('#define X(x) \\\n(x)\n') == hook.lexical_signature('#define X(x) (x)\n')
    count = changed = 0
    for path in sorted((ROOT / 'docs').rglob('*.md')):
        if 'validation' in path.relative_to(ROOT / 'docs').parts:
            continue
        original = path.read_text()
        markdown = original
        if (ROOT / 'hooks/reader_comments.py').exists():
            relative = path.relative_to(ROOT / 'docs').as_posix()
            markdown = load('reader_comments').on_page_markdown(
                markdown, SimpleNamespace(file=SimpleNamespace(src_path=relative)))
        shown = hook.on_page_markdown(markdown)
        assert hook.on_page_markdown(shown) == shown, 'Non-idempotent formatting: ' + str(path)
        expected = []
        for before, after in zip(hook.FENCES.finditer(markdown), hook.FENCES.finditer(shown)):
            if before[2].strip() in hook.LANGUAGES:
                assert hook.lexical_signature(before[3]) == hook.lexical_signature(after[3])
                expected.append(after[3])
                count += 1
                changed += before[3] != after[3]
            else:
                assert before[0] == after[0], 'Non-source block changed: ' + str(path)
        assert original == path.read_text(), 'Stored source changed: ' + str(path)
        if site and expected:
            relative = path.relative_to(ROOT / 'docs')
            target = site / relative.parent / ('' if relative.stem == 'index' else relative.stem) / 'index.html'
            parser = Listings(); parser.feed(target.read_text())
            assert parser.blocks == expected, 'Built source differs from the checked display: ' + str(target)
    print(f'PASS: {count} source listings; {changed} reformatted; tokens/comments preserved, idempotent, stored sources unchanged' + ('; built HTML matches.' if site else '.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-dir', type=Path)
    args = parser.parse_args()
    verify(args.site_dir)
