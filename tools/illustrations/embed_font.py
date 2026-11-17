"""Build one reusable, licensed, self-contained Inter subset for the new art."""
from pathlib import Path
from io import BytesIO
import base64,re
from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
ROOT=Path(__file__).resolve().parents[2]
font=TTFont(ROOT/'tools/assets/Inter.ttf')
font=instantiateVariableFont(font,{'opsz':14},inplace=True)
chars=''.join(p.read_text() for p in (ROOT/'tools/illustrations').glob('*.py'))
options=subset.Options();options.flavor='woff2'
sub=subset.Subsetter(options=options);sub.populate(text=chars);sub.subset(font)
font.flavor='woff2';font.recalcTimestamp=False
out=BytesIO();font.save(out)
encoded=base64.b64encode(out.getvalue()).decode()
style=f'<style type="text/css">@font-face{{font-family:BookDiagram;src:url(data:font/woff2;base64,{encoded}) format("woff2");font-style:normal;font-weight:100 900}} text{{font-family:BookDiagram,Inter,Arial,sans-serif;font-synthesis:none;text-rendering:optimizeLegibility}}</style>'
(ROOT/'tools/illustrations/font-style.svg').write_text(style+'\n')
print(f'Embedded Inter subset: {len(out.getvalue())} bytes.')
