"""Preserve recently published chapter URLs after the curriculum was sequenced."""
from html import escape
import json
import re
REDIRECTS = {'16-cutlass-gemm.md': '24-cutlass-gemm/', '17-cutlass-pipelines.md': '25-cutlass-pipelines/', '27-benchmarking-integration.md': '28-benchmarking-integration/', '28-workbook-capstone.md': '29-workbook-capstone/', '17-cute-tensors-tiling.md': '18-cute-tensors-tiling/', '19-tiled-copy.md': '20-tiled-copy/', '18-cta-indexing.md': '19-cta-indexing/', '26-precision-block-scaling.md': '27-precision-block-scaling/', '21-shared-layouts.md': '22-shared-layouts/', '24-cutlass-pipelines.md': '25-cutlass-pipelines/', '25-epilogue-fusion.md': '26-epilogue-fusion/', '23-cutlass-gemm.md': '24-cutlass-gemm/', '16-cute-layout-algebra.md': '17-cute-layout-algebra/', '20-mma-atoms.md': '21-mma-atoms/', '22-gemm-construction.md': '23-gemm-construction/'}
def on_post_page(output, page, config, **kwargs):
    target = REDIRECTS.get(page.file.src_path)
    if target is None: return output
    relative = '../'+target
    canonical = config['site_url'].rstrip('/')+'/'+target
    old_number=re.match(r'(\d+)',page.file.src_path)[1]
    new_number=re.match(r'(\d+)',target)[1]
    # Preserve section links, including headings whose chapter number changed.
    script=('let h=location.hash;'
            'h=h.replace('+json.dumps('#lesson-'+old_number+'-')+','+json.dumps('#lesson-'+new_number+'-')+');'
            'h=h.replace('+json.dumps('#'+old_number+'-')+','+json.dumps('#'+new_number+'-')+');'
            'location.replace('+json.dumps(relative)+'+h);')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<script>'+script+'</script>'
            '<meta http-equiv="refresh" content="0; url='+escape(relative)+'">'
            '<link rel="canonical" href="'+escape(canonical)+'">'
            '<title>Chapter moved</title></head><body><p>This chapter has moved. '
            '<a href="'+escape(relative)+'">Continue to the renumbered chapter.</a>'
            '</p></body></html>')
