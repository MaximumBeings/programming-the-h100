#!/usr/bin/env python3
"""Reproduce the original homepage reading-map SVG with embedded, licensed type."""
from pathlib import Path
from illustrations.drawing import Drawing, ROOT, COLORS, MUTED

output = ROOT/'docs/images/book'
output.mkdir(exist_ok=True)
d = Drawing('book/overview.svg', 'From the machine to a measured kernel',
            'A reading map for all 28 chapters: place data, establish ownership, overlap work, then measure.',
            'Reading map for Programming the H100 · schematic dependencies, not a hardware block diagram')
d.parts = [part.replace('CHAPTER ok   /   FIGURE overview.svg', 'THE WHOLE BOOK   /   READING MAP') for part in d.parts]
y=d.y

def whole(title, body, color):
    global y
    h=d.cardheight(title,body,704,size=22)
    d.card(28,y,704,title,body,color,h,size=22)
    y+=h

def connector():
    global y
    d.line(380,y+4,380,y+27,MUTED,2,True)
    y+=36

def pair(left,right):
    global y
    h=max(d.cardheight(t,b,342,size=22) for t,b,c in (left,right))
    for x,(title,body,color) in zip((28,390),(left,right)):
        d.card(x,y,342,title,body,color,h,size=22)
    y+=h

whole('01–02  Understand the machine',
      'SMs, memory, clusters and PTX: where does each value live?', 'blue')
connector()
whole('03  Make asynchronous work safe',
      'Track completion, publish writes and prevent premature buffer reuse.', 'amber')
connector()
pair(('04–05  Move the tiles','Tensor maps, TMA, shared-memory layout and swizzling.','teal'),
     ('06–07  Compute on tiles','WGMMA, register fragments, groups and epilogues.','violet'))
connector()
whole('08–10  Compose a kernel',
      'Warp specialization, pipelines, persistent work and Stream-K scheduling.', 'teal')
connector()
pair(('11  Scale across GPUs','Topology, NCCL and hybrid parallel training.','blue'),
     ('12  Apply it to attention','Online softmax and an annotated FA3 kernel design.','violet'))
connector()
whole('13–14  Measure, explain, refine',
      'Use Systems for the application timeline and Compute for the selected kernel.', 'amber')
connector()
whole('15–18  Introductory CuTe',
      'Layout algebra, tensor views, predicates and complete coordinate-to-byte indexing.', 'blue')
connector()
whole('19–23  Intermediate kernels',
      'Thread ownership, copy and MMA atoms, shared layouts and checked GEMM implementations.', 'teal')
connector()
whole('24–28  Advanced CUTLASS',
      'Pipelines, fusion, precision, integration and a measured capstone with worked labs.', 'violet')
d.y=y+24
d.note('The invariant at every layer',
       'Know who owns each value, when it is ready, when its storage is reusable, and how the result is checked.', 'gray')
height=d.write()
# Give the accessible description the same useful information as the visual map.
p=output/'overview.svg'
s=p.read_text();a=s.index('<desc id="figure-desc">');b=s.index('</desc>',a)
s=s[:a]+'<desc id="figure-desc">Chapters 1–2: hardware, memory and PTX. Chapter 3: asynchronous completion and ownership. Chapters 4–5: TMA and layouts; chapters 6–7: WGMMA and epilogues. Chapters 8–10 combine these in pipelined kernels. Chapter 11 scales across GPUs; chapter 12 applies the ideas to attention. Chapters 13–14 use kernel and application profiling to guide another iteration. Chapters 15–28 progress from introductory CuTe layouts and indexing through complete copy/GEMM kernels to advanced CUTLASS pipelines, fusion, precision and reproducible integration. At every layer, check ownership, readiness, reuse and correctness.'+s[b:]
p.write_text(s)
print(f'Created {p.relative_to(ROOT)} (760 × {height}).')
