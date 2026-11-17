#!/usr/bin/env python3
"""Draw the three-engine overview with legible labels in a compact canvas."""
from pathlib import Path
from html import escape
parts=[]
def text(x,y,value,size=18,color='#172033',weight=450,anchor='middle'):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')
def box(x,y,w,h,fill='#fff',stroke='#cbd5e1'):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
def arrow(x1,y1,x2,y2):
    parts.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="#087f75" stroke-width="2" marker-end="url(#arrow)"/>')
text(360,40,'H100 SM — Three Async Engines',29,weight=650)
text(360,74,'Independent work can overlap; completion is explicit.',18,'#475569')
box(16,100,688,405,'#f8fafc')
text(360,135,'STREAMING MULTIPROCESSOR',17,'#475569',650)
for x,title,subtitle,lines in [
    (36,'Warp schedulers','Instruction issue',['Schedule warps','and issue their','instructions']),
    (270,'Tensor Cores','WGMMA math',['Asynchronous','matrix operations','on data tiles']),
    (504,'TMA engine','Data movement',['Asynchronous','tensor transfers','HBM → SMEM']),
]:
    box(x,160,180,225,'#e8f6f3','#087f75')
    text(x+90,195,title,20,'#065f57',650)
    text(x+90,224,subtitle,17,'#475569')
    for i,line in enumerate(lines):text(x+90,272+i*27,line,18)
box(36,411,414,67,'#eaf0ff','#8ba5d6')
text(243,437,'WGMMA completion',18,'#244a87',650)
text(243,464,'commit_group → wait_group',17,'#244a87')
box(474,411,210,67,'#e8f6f3','#087f75')
text(579,437,'TMA completion',18,'#065f57',650)
text(579,464,'mbarrier',20,'#065f57',650)
arrow(360,385,360,407)
arrow(594,385,594,407)
text(360,545,'Overlap does not remove synchronization.',20,weight=650)
text(360,576,'Observe the correct completion protocol before using results.',17,'#475569')
source='https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html'
text(360,612,'Conceptual overview · NVIDIA PTX ISA 8.7: WGMMA and TMA completion',14,'#475569')
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="640" viewBox="0 0 720 640" role="img" aria-labelledby="title desc">
<title id="title">H100 SM — three independent async units</title>
<desc id="desc">Warp schedulers issue instructions, Tensor Cores perform matrix math, and TMA transfers tensors. Work may overlap. WGMMA completion is observed through commit and wait groups; TMA completion can use mbarrier. Source: {source}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#087f75"/></marker></defs><rect width="720" height="640" fill="#fff"/>
{''.join(parts)}</svg>'''
p=Path(__file__).resolve().parents[1]/'docs/images/ch03/04-sm-async-units.svg'
p.write_text(svg)
