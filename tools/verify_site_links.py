#!/usr/bin/env python3
"""Check built-page IDs, internal file targets, and fragment links offline."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.references = []
        self.duplicates = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                self.duplicates.append(attrs['id'])
            self.ids.add(attrs['id'])
        for name in ('href', 'src'):
            if name in attrs:
                self.references.append(attrs[name])


def verify(site, site_url):
    if not site.is_dir():
        print(f'FAIL: built site does not exist: {site}; run mkdocs build --strict first.')
        return 1
    base = urlsplit(site_url)
    prefix = base.path.rstrip('/') + '/'
    pages = {}
    for path in site.rglob('*.html'):
        page = Page()
        page.feed(path.read_text())
        pages[path.resolve()] = page
    if not pages:
        print('FAIL: no built HTML pages were found.')
        return 1
    errors, checked = [], 0
    for path, page in pages.items():
        for ident in page.duplicates:
            errors.append(f'{path.relative_to(site)}: duplicate id {ident!r}')
        for href in page.references:
            url = urlsplit(href)
            if url.scheme and url.scheme not in ('http', 'https'):
                continue
            if url.netloc and (url.netloc != base.netloc or not url.path.startswith(prefix)):
                continue
            decoded = unquote(url.path)
            if decoded.startswith(prefix):
                target = site / decoded[len(prefix):]
            elif decoded.startswith('/'):
                target = site / decoded.lstrip('/')
            elif decoded:
                target = path.parent / decoded
            else:
                target = path
            if target.is_dir():
                target /= 'index.html'
            target = target.resolve()
            checked += 1
            if not target.is_relative_to(site):
                errors.append(f'{path.relative_to(site)}: link escapes built site: {href}')
            elif not target.is_file():
                errors.append(f'{path.relative_to(site)}: missing target: {href}')
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.append(f'{path.relative_to(site)}: missing fragment: {href}')
    for error in errors[:100]:
        print('FAIL:', error)
    if len(errors) > 100:
        print(f'FAIL: {len(errors)-100} additional errors omitted.')
    if errors:
        return 1
    print(f'PASS: {len(pages)} HTML pages; {checked} internal targets/fragments; no duplicate IDs.')
    return 0


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-dir', type=Path, default=root / 'site')
    parser.add_argument('--site-url', default='https://maximumbeings.github.io/programming-the-h100/')
    args = parser.parse_args()
    return verify(args.site_dir.resolve(), args.site_url)


if __name__ == '__main__':
    raise SystemExit(main())
