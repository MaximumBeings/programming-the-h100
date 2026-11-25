#!/usr/bin/env python3
"""Verify source-listing equality, Python syntax and the copyable built HTML."""
import argparse
import ast
from html.parser import HTMLParser
from pathlib import Path
from embed_inference_sources import pages, expected_page, write_bundle


class PythonListings(HTMLParser):
    def __init__(self):
        super().__init__();self.divs=[];self.pre=0;self.parts=None;self.blocks=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='div':self.divs.append(set(attrs.get('class','').split()))
        if tag=='pre':self.pre+=1
        if tag=='code' and self.pre:self.parts=[]
    def handle_data(self,data):
        if self.parts is not None:self.parts.append(data)
    def handle_endtag(self,tag):
        if tag=='code' and self.parts is not None:
            if any('language-python' in classes for classes in self.divs):self.blocks.append(''.join(self.parts))
            self.parts=None
        if tag=='pre':self.pre-=1
        if tag=='div':self.divs.pop()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--site-dir',type=Path);args=ap.parse_args();count=0
    for n,page,files in pages():
        assert page.read_text()==expected_page(n,page,files),f'source listing drift: {page.name}'
        parsed=PythonListings()
        if args.site_dir:
            parsed.feed((args.site_dir/page.stem/'index.html').read_text())
        for source in files:
            expected=source.read_text().rstrip()+'\n';ast.parse(expected)
            if args.site_dir:
                assert expected in parsed.blocks,f'built Python listing changed or missing: {source.name}'
            count+=1
    write_bundle(check=True)
    assert count==18
    print('PASS: all 18 complete Python listings, syntax and downloadable bundle match'+('; rendered copyable code matches exact source' if args.site_dir else ''))

if __name__=='__main__':main()
