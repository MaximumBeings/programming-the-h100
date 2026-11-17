"""Keep SVG labels at least 15.4 CSS pixels (about 11.5 points) when a figure is embedded.

Wide technical figures pan inside their existing image links on narrow screens,
so responsive shrinking cannot undo the SVG's minimum font size.
"""
from pathlib import Path
import re
import hashlib

SIZE = re.compile(r'font-size(?:\s*:\s*|=")([\d.]+)')
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
        # Opening illustrations retain their original responsive banner size.
        if relative.startswith('images/chapter-openers/'):
            return m[0]
        vb=re.search(r'viewBox="[\d.-]+ [\d.-]+ ([\d.]+) [\d.]+"',source)
        sizes=[float(s) for s in SIZE.findall(source) if float(s)>0]
        if not vb or not sizes:return m[0]
        minimum=float(vb[1])*15.4/min(sizes)
        return m[0].replace('<img ',f'<img style="min-width: {minimum:.2f}px" ',1)
    output=SVG.sub(image,output)
    for original,versioned in versions.items():
        output=output.replace('src="'+original+'"','src="'+versioned+'"').replace('href="'+original+'"','href="'+versioned+'"')
    return output
