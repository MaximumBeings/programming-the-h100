"""Preserve recently published chapter URLs after the curriculum was sequenced."""
from html import escape
REDIRECTS = {'16-cutlass-gemm.md':'23-cutlass-gemm/',
             '17-cutlass-pipelines.md':'24-cutlass-pipelines/'}
def on_post_page(output, page, config, **kwargs):
    target = REDIRECTS.get(page.file.src_path)
    if target is None: return output
    relative = '../'+target
    canonical = config['site_url'].rstrip('/')+'/'+target
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="refresh" content="0; url='+escape(relative)+'">'
            '<link rel="canonical" href="'+escape(canonical)+'">'
            '<title>Chapter moved</title></head><body><p>This chapter has moved. '
            '<a href="'+escape(relative)+'">Continue to the renumbered chapter.</a>'
            '</p></body></html>')
