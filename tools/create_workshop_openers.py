#!/usr/bin/env python3
"""Matching conceptual chapter artwork for the CuTe/CUTLASS course."""
from pathlib import Path
from html import escape
from create_chapter_openers import Drawing, TEAL, GOLD, GREEN, MUTED
from illustrations.drawing import FONT_STYLE, LICENSE
ROOT=Path(__file__).resolve().parents[1]
SPECS=[
(14,'See the','whole application','CPU launch · copies · kernel timelines',['CPU','COPY','KERNEL']),
(15,'Start with','a coordinate.','Shapes · strides · physical offsets',['SHAPE','STRIDE','OFFSET']),
(16,'Trace the stride.','Find the value.','Logical grids · physical storage · byte addresses',['COORDINATE','STRIDE','ADDRESS']),
(17,'Compose shapes.','Prove the map.','Nested modes · composition · complement',['DOMAIN','LAYOUT','IMAGE']),
(18,'Views are','not transfers.','Tensor engines · views · predicates',['POINTER','VIEW','TILE']),
(19,'Follow every','address.','CTA origins · operand strides · edges',['CTA','ORIGIN','ADDRESS']),
(20,'One value.','One owner.','Thread maps · value maps · copies',['THREAD','VALUE','COPY']),
(21,'Know the','MMA fragments.','Instruction atoms · lanes · registers',['LANE','REGISTER','OUTPUT']),
(22,'Rearrange data.','Respect banks.','Shared-memory layouts · bank mapping',['ADDRESS','BANK','ACCESS']),
(23,'Build the','whole reduction.','Shared tiles · SIMT math · guarded stores',['LOAD','REDUCE','STORE']),
(24,'Assemble','a checked GEMM.','Collectives · kernel · device adapter',['MAINLOOP','EPILOGUE','ADAPTER']),
(25,'Overlap work.','Protect the stage.','Producer · consumer · safe reuse',['ACQUIRE','READY','RELEASE']),
(26,'Fuse the graph.','Preserve math.','Linear combination · activation · store',['αAB + βC','ReLU','STORE']),
(27,'Pack fewer bits.','Keep the contract.','E2M1 values · scales · numerical checks',['PACK','SCALE','DECODE']),
(28,'Measure a scope.','Explain results.','Correctness · timing · profiler evidence',['VERIFY','MEASURE','EXPLAIN']),
(29,'Prove it.','Then improve it.','References · kernels · reproducible evidence',['REFERENCE','KERNEL','EVIDENCE'])]

def art(n,labels):
    d=Drawing()
    # The tile motif follows the chapter's ownership or dataflow model.
    if n==14:
        for r,label in enumerate(['CPU','STREAM 0','STREAM 1']):
            d.text(10,75+r*100,label,21,MUTED)
            d.line(145,70+r*100,780,70+r*100,MUTED,1)
            for j in range(3):
                x=160+j*190+(r%2)*55
                d.box(x,45+r*100,140,50,['LAUNCH','COPY','KERNEL'][(j+r)%3],TEAL if j%2 else GOLD,size=20)
        d.text(400,400,'Read the application before tuning one kernel',23,MUTED,'middle')
    elif n in (15,16,17,18,19,20,21,22,27):
        d.text(180,47,labels[0],24,GREEN,'middle','700')
        d.text(600,47,labels[2],24,GOLD,'middle','700')
        d.grid(55,90,4,6,43,TEAL)
        d.grid(475,90,4,6,43,GOLD)
        d.line(325,175,455,175,TEAL,3,True)
        d.box(335,215,115,55,labels[1],TEAL,size=18)
        if n==21:
            d.text(180,315,'32 lanes × 4 accumulators',22,MUTED,'middle')
            d.text(600,315,'16 × 8 output values',22,MUTED,'middle')
        elif n==27:
            d.text(180,315,'Two E2M1 codes per byte',22,MUTED,'middle')
            d.text(600,315,'Values × defined block scales',22,MUTED,'middle')
        else:
            d.text(180,315,'Logical coordinates',23,MUTED,'middle')
            d.text(600,315,'Explicit storage mapping',23,MUTED,'middle')
        d.text(400,390,'Enumerate a small case. Check every destination.',22,MUTED,'middle')
    else:
        for i,label in enumerate(labels):
            x=10+i*270
            d.box(x,85,240,220,color=TEAL if i!=1 else GOLD)
            d.grid(x+52,115,3,4,35,GOLD if i==1 else TEAL)
            d.text(x+120,275,label,22,GOLD if i==1 else TEAL,'middle','700')
            if i<2:d.line(x+243,195,x+265,195,TEAL,3,True)
        if n==25:
            d.line(680,330,140,330,GOLD,2,True)
            d.text(400,385,'Reuse only after the consumer releases ownership',23,MUTED,'middle')
        else:d.text(400,385,'Make the contract explicit at every boundary',23,MUTED,'middle')
    return ''.join(d.parts)

def main():
    for n,a,b,subtitle,labels in SPECS:
        svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="520" viewBox="0 0 1440 520" role="img" aria-labelledby="title description"><title id="title">Chapter {n}: {escape(a+' '+b)}</title><desc id="description">{escape(subtitle)}. Conceptual illustration, not a measured trace or hardware diagram.</desc>{LICENSE}<defs>{FONT_STYLE}<linearGradient id="bg" x2="1" y2="1"><stop stop-color="#071a20"/><stop offset="1" stop-color="#0e3033"/></linearGradient><pattern id="dots" width="28" height="28" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#9ccfc0" opacity=".16"/></pattern><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="none" stroke="#2dd4bf" stroke-width="1.5"/></marker></defs><rect width="1440" height="520" rx="20" fill="url(#bg)"/><rect width="1440" height="520" rx="20" fill="url(#dots)"/><g font-family="Inter, Arial, sans-serif"><rect x="55" y="62" width="5" height="315" rx="2" fill="#a3e635"/><text x="83" y="94" font-size="19" letter-spacing="3" fill="#a3e635" font-weight="700">CHAPTER {n:02}</text><text x="83" y="176" font-size="43" fill="#eef9f5" font-weight="700">{escape(a)}</text><text x="83" y="232" font-size="43" fill="#eef9f5" font-weight="700">{escape(b)}</text>'''
        for i,line in enumerate(subtitle.split(' · ')):
            svg+=f'<text x="83" y="{290+i*31}" font-size="21" fill="#a8c6c8">{escape(line)}</text>'
        svg+=f'<text x="83" y="440" font-size="18" fill="#a8c6c8">PROGRAMMING THE H100</text><g transform="translate(555 30)">{art(n,labels)}</g><text x="1390" y="490" text-anchor="end" font-size="15" fill="#89b0b2">CONCEPTUAL ILLUSTRATION</text></g></svg>\n'
        (ROOT/f'docs/images/chapter-openers/{n:02}-workshop.svg').write_text(svg)
    print(f'Created {len(SPECS)} matching workshop chapter openers.')
if __name__=='__main__':main()
