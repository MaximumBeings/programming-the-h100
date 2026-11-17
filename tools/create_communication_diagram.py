#!/usr/bin/env python3
"""Draw the chapter 2 communication comparison. Run polish_diagrams.py afterward."""
from pathlib import Path
from html import escape
parts=[]
def text(x,y,s,size=22,fill='#152536',weight=450,anchor='start'):
 parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>')
def rect(x,y,w,h,fill='#fff',stroke='#d9e3ea',r=16):
 parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
def arrow(x1,y1,x2,y2,color='#b45342'):
 parts.append(f'<path d="M{x1} {y1} H{x2}" fill="none" stroke="{color}" stroke-width="3" marker-end="url(#arrow-{color[1:]})"/>')
def node(x,y,title,lines,accent):
 rect(x,y,228,116,stroke='#c7d7df')
 parts.append(f'<rect x="{x}" y="{y+18}" width="4" height="80" rx="2" fill="{accent}"/>')
 text(x+22,y+34,title,24,weight=650)
 for i,line in enumerate(lines):text(x+22,y+65+i*25,line,20,'#475569')
text(42,43,'HOPPER / THREAD BLOCK CLUSTERS',15,'#526d80',650)
text(42,88,'Sharing results between blocks',34,weight=650)
text(42,124,'Block A produces a partial sum. Block B needs to consume it.',21,'#475569')
rect(32,158,1036,278,'#fcf9f7','#eaded8',20)
text(56,195,'01',18,'#ad5140',650)
text(96,195,'WITHOUT CLUSTERS',22,'#8c392c',650)
text(56,225,'Exchange through a global-memory buffer',20,'#586473')
node(56,250,'Block A / SM 0',['Compute partial sum','Store global buffer'],'#b45342')
node(816,250,'Block B / SM 1',['Load global buffer','Use the result'],'#b45342')
rect(435,250,230,116,'#f4eee9','#decac0')
text(550,285,'Global buffer',23,weight=650,anchor='middle')
text(550,314,'L2 cache / HBM path',19,'#475569',anchor='middle')
text(550,340,'HBM traffic on misses',17,'#586473',anchor='middle')
arrow(298,309,421,309)
arrow(679,309,802,309)
text(360,284,'STORE',15,'#8c392c',650,'middle')
text(740,284,'LOAD',15,'#8c392c',650,'middle')
text(56,406,'Coordinate producer completion and visibility before Block B reads.',20,'#6b473b')
rect(32,458,1036,278,'#f3faf8','#c8e3dc',20)
text(56,495,'02',18,'#08776d',650)
text(96,495,'WITH CLUSTERS',22,'#08675f',650)
text(56,525,'Exchange through distributed shared memory (DSMEM)',20,'#48665f')
node(56,550,'Block A / SM 0',['Compute partial sum','Keep it in A’s SMEM'],'#087f75')
node(816,550,'Block B / SM 1',['Read A’s SMEM','Use the result'],'#087f75')
arrow(298,610,802,610,'#087f75')
text(550,585,'DSMEM ACCESS',19,'#08675f',650,'middle')
text(550,644,'Within the same thread block cluster',17,'#48665f',anchor='middle')
text(56,706,'Synchronize the cluster; keep Block A’s shared memory alive until reads finish.',20,'#285b52')
text(42,777,'KEY DIFFERENCE',15,'#526d80',650)
text(42,809,'The DSMEM exchange avoids staging the partial sum through global memory.',21,weight=550)
text(42,843,'Conceptual data paths. Latency and speedup depend on the workload; this is not a benchmark.',17,'#526d80')
markers=''.join(f'<marker id="arrow-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="{c}"/></marker>' for c in ['#b45342','#087f75'])
s=f'''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="875" viewBox="0 0 1100 875" role="img" aria-labelledby="title desc">
<title id="title">The communication problem — with and without clusters</title><desc id="desc">Two aligned panels compare global-memory exchange between blocks with DSMEM access inside a Hopper thread block cluster. Both paths require correct synchronization; DSMEM also requires remote shared-memory lifetime management. No fixed latency or speedup is asserted.</desc><defs>{markers}</defs><rect width="1100" height="875" fill="#fff"/>
{''.join(parts)}</svg>'''
(Path(__file__).resolve().parents[1] / 'docs/images/ch02/02-communication-with-without-clusters.svg').write_text(s)
