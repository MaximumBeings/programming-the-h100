"""Version SVG URLs so browsers load the current diagram assets.

Figures fit the reading column; typography is controlled inside each SVG.
"""
from pathlib import Path
import re
import hashlib

SVG = re.compile(r'<img\b[^>]*\bsrc="([^"]+\.svg)"[^>]*>')

def on_post_page(output, page, config, **kwargs):
    versions={}
    def image(m):
        url = m[1]
        # Resolve relative to the generated page URL, then use its images tail.
        if '/images/' in url:relative='images/'+url.split('/images/',1)[1]
        elif url.startswith('images/'):relative=url
        else:return m[0]
        p=Path(config['docs_dir'])/relative
        if not p.is_file():return m[0]
        source=p.read_text()
        versions[url]=url+'?v='+hashlib.sha256(source.encode()).hexdigest()[:12]
        return m[0]
    output=SVG.sub(image,output)
    for original,versioned in versions.items():
        output=output.replace('src="'+original+'"','src="'+versioned+'"').replace('href="'+original+'"','href="'+versioned+'"')
    return output
