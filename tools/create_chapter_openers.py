#!/usr/bin/env python3
"""Create original, schematic SVG chapter illustrations with selectable text."""
from html import escape
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs' / 'images' / 'chapter-openers'
OUTPUT.mkdir(parents=True, exist_ok=True)
GREEN, TEAL, GOLD, INK, MUTED = '#a3e635', '#2dd4bf', '#fbbf24', '#07191c', '#a8c6c8'


class Drawing:
    def __init__(self):
        self.parts = []

    def box(self, x, y, w, h, label='', color=TEAL, fill='#102c32', size=23):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{color}" stroke-width="2"/>')
        if label:
            self.text(x+w/2, y+h/2+size*.34, label, size, color, 'middle')

    def text(self, x, y, value, size=23, color='#e8f5f3', anchor='start', weight='400'):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>')

    def line(self, x1, y1, x2, y2, color=TEAL, width=3, arrow=False):
        self.parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{color}" stroke-width="{width}" fill="none"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')

    def grid(self, x, y, rows=4, cols=4, cell=24, color=TEAL, triangular=False):
        for r in range(rows):
            for c in range(cols):
                enabled = not triangular or c <= r
                self.parts.append(f'<rect x="{x+c*cell}" y="{y+r*cell}" width="{cell-4}" height="{cell-4}" rx="3" fill="{color if enabled else "#1a333a"}" opacity="{0.8 if enabled else 0.4}"/>')

    def chip(self, x, y, label, color=TEAL, w=150, h=140):
        self.box(x, y, w, h, color=color)
        self.grid(x+17, y+18, 3, 4, min(27, (w-30)/4), color)
        self.text(x+w/2, y+h-17, label, 23, color, 'middle', '700')
        for i in range(6):
            yy = y+18+i*(h-36)/5
            self.line(x-9, yy, x, yy, color, 3)
            self.line(x+w, yy, x+w+9, yy, color, 3)


def illustration(kind):
    d = Drawing()
    if kind == 'evolution':
        for i, (name, year, gpu) in enumerate([('Volta','2017','V100'),('Ampere','2020','A100'),('Hopper','2022','H100'),('Blackwell','2024','B200')]):
            x = i*200+10
            d.text(x+75, 40, name, 25, GOLD if i == 2 else '#e8f5f3', 'middle', '700')
            d.chip(x, 80, gpu, GOLD if i == 2 else TEAL)
            d.text(x+75, 270, year, 25, MUTED, 'middle')
        d.line(20, 308, 770, 308, TEAL, 2, True)
        d.text(400, 358, 'Architecture introduction years · illustrative packages', 20, MUTED, 'middle')
    elif kind == 'clusters':
        for group in range(2):
            x=15+group*410
            d.box(x, 45, 370, 270, color=TEAL if group == 0 else GOLD)
            d.text(x+185, 85, f'Cluster {group}', 25, '#e8f5f3', 'middle', '700')
            d.chip(x+35, 115, 'SM / CTA', TEAL, 130, 130)
            d.chip(x+205, 115, 'SM / CTA', TEAL, 130, 130)
            d.line(x+165, 183, x+205, 183, GOLD, 4)
            d.text(x+185, 285, 'Distributed shared memory', 21, MUTED, 'middle')
        d.text(400, 370, 'Co-scheduled blocks · explicit cluster synchronization', 23, MUTED, 'middle')
    elif kind == 'barriers':
        d.box(20, 90, 200, 170, 'PRODUCER', TEAL)
        d.box(320, 110, 180, 130, 'STAGE 0', GOLD)
        d.box(600, 90, 200, 170, 'CONSUMER', GREEN)
        d.line(220, 150, 320, 150, TEAL, 5, True)
        d.line(500, 150, 600, 150, TEAL, 5, True)
        d.text(410, 65, 'FULL · data is ready', 25, GOLD, 'middle')
        d.line(700, 290, 120, 290, GOLD, 3, True)
        d.text(410, 342, 'EMPTY · all readers have finished', 25, MUTED, 'middle')
    elif kind == 'descriptor':
        d.box(5, 75, 240, 230, color=GOLD)
        d.text(125, 120, 'TENSOR MAP', 26, GOLD, 'middle', '700')
        for i, value in enumerate(['shape','byte strides','tile extent','swizzle']):
            d.text(35, 164+i*35, value, 23, MUTED)
        d.line(245, 190, 315, 190, TEAL, 4, True)
        d.grid(335, 70, 8, 8, 32, TEAL)
        d.text(461, 360, 'Logical tile', 24, '#e8f5f3', 'middle')
        d.line(600, 190, 650, 190, GOLD, 4, True)
        for row in range(8):
            for col in range(4):
                d.box(675+col*30, 70+row*32, 24, 25, color=GOLD if col == row%4 else TEAL)
        d.text(735, 360, 'Bank layout', 24, '#e8f5f3', 'middle')
    elif kind == 'tma':
        for i in range(3):
            d.box(20+i*12, 55+i*30, 160, 200, color=TEAL, fill='#123840')
        d.text(120, 310, 'HBM', 27, TEAL, 'middle', '700')
        d.box(300, 100, 190, 140, 'TMA', GOLD, size=37)
        d.line(205, 170, 300, 170, TEAL, 7, True)
        d.line(490, 170, 600, 170, TEAL, 7, True)
        d.grid(620, 70, 6, 6, 30, TEAL)
        d.text(700, 310, 'Shared memory', 25, TEAL, 'middle')
        d.box(295, 310, 200, 55, 'Compute continues', GREEN, size=20)
    elif kind == 'wgmma':
        d.grid(0, 100, 4, 4, 27, TEAL); d.text(50, 80, 'A', 29, TEAL, 'middle')
        d.grid(145, 100, 4, 4, 27, GOLD); d.text(195, 80, 'B', 29, GOLD, 'middle')
        d.line(260, 160, 320, 160, TEAL, 4, True)
        d.chip(345, 70, 'TENSOR CORES', GREEN, 230, 190)
        d.line(590, 160, 650, 160, TEAL, 4, True)
        d.grid(680, 100, 4, 4, 27, GREEN); d.text(730, 80, 'D', 29, GREEN, 'middle')
        for i in range(4):
            d.box(170+i*120, 320, 104, 43, f'Warp {i}', TEAL, size=20)
        d.text(410, 405, 'One 128-thread WGMMA collective', 24, MUTED, 'middle')
    elif kind == 'groups':
        for row, color in enumerate([TEAL, GOLD]):
            y=75+row*110
            d.text(0, y+35, f'Group {row}', 24, color)
            for i in range(4):d.box(125+i*110, y, 95, 65, 'MMA', color, size=21)
            d.line(570, y+33, 635, y+33, color, 4, True)
            d.grid(660, y-2, 2, 4, 29, color)
        d.line(600, 35, 600, 315, GREEN, 3)
        d.text(600, 355, 'Commit → wait → complete epilogue', 25, MUTED, 'middle')
    elif kind == 'specialization':
        for lane, label, color in [(0,'PRODUCER',GOLD),(1,'CONSUMER 0',TEAL),(2,'CONSUMER 1',GREEN)]:
            y=55+lane*110
            d.text(0, y+39, label, 22, color)
            for i in range(4):d.box(205+i*145, y, 125, 70, 'LOAD' if lane == 0 else 'MMA', color, size=24)
        d.text(480, 405, 'Different roles · overlapping work', 26, MUTED, 'middle')
    elif kind == 'ring':
        points=[(405,45),(610,260),(200,260)]
        for i,(x,y) in enumerate(points):
            nx,ny=points[(i+1)%3];d.line(x+35,y+45,nx+35,ny+45,TEAL,4,True)
        for i,(x,y) in enumerate(points):d.box(x-70,y,190,90,f'STAGE {i}',[GOLD,TEAL,GREEN][i],size=27)
        d.box(20,75,190,70,'Producer',GOLD);d.box(600,75,190,70,'Consumer',GREEN)
        d.text(420,210,'Reuse only after',25,MUTED,'middle')
        d.text(420,243,'readers finish',25,MUTED,'middle')
        d.text(420,405,'Phase tracking · ownership · safe reuse',24,MUTED,'middle')
    elif kind == 'streamk':
        d.text(400,35,'Partition the reduction work',27,'#e8f5f3','middle','700')
        colors=[TEAL,GOLD,GREEN,'#93c5fd']
        for i in range(24):d.box(20+i*32,65,26,65,color=colors[i//6])
        for i,color in enumerate(colors):
            x=40+i*200
            d.line(110+i*190,145,x+65,205,color,3,True)
            d.chip(x,215,f'CTA {i}',color,140,130)
        d.text(420,405,'Balance K work · combine partial results',24,MUTED,'middle')
    elif kind == 'fabric':
        positions=[]
        for i in range(8):
            angle=math.pi*2*i/8
            positions.append((400+320*math.cos(angle),195+145*math.sin(angle)))
        for x,y in positions:d.line(400,195,x,y,TEAL,3)
        for i,(x,y) in enumerate(positions):d.box(x-48,y-30,96,60,f'GPU {i}',TEAL,size=20)
        d.box(310,145,180,100,'NVSwitch',GOLD,size=29)
        d.text(400,420,'Topology → collective → distributed workload',24,MUTED,'middle')
    elif kind == 'attention':
        d.grid(10,70,8,8,29,TEAL,True)
        d.text(120,345,'Causal attention',23,TEAL,'middle')
        for i,label in enumerate(['Q','K','V']):d.box(315,75+i*90,80,65,label,GOLD,size=29)
        d.line(405,180,460,180,TEAL,4,True)
        d.box(475,110,170,140,color=GREEN)
        d.text(560,165,'Online',27,GREEN,'middle','700');d.text(560,201,'softmax',27,GREEN,'middle','700')
        d.line(650,180,700,180,TEAL,4,True);d.grid(715,120,4,3,29,GREEN)
        d.text(425,410,'Keep intermediates on chip',25,MUTED,'middle')
    elif kind == 'profile':
        d.line(50,325,50,40,MUTED,2);d.line(50,325,780,325,MUTED,2)
        d.line(50,325,360,85,TEAL,5);d.line(360,85,760,85,GREEN,5)
        d.parts.append('<circle cx="275" cy="210" r="12" fill="#fbbf24"/><circle cx="275" cy="210" r="24" fill="none" stroke="#fbbf24" opacity=".4"/>')
        d.text(305,228,'Measure first',26,GOLD)
        d.text(535,60,'Compute ceiling',23,GREEN,'middle')
        d.text(390,375,'Counters · SASS · occupancy · evidence',25,MUTED,'middle')
    elif kind == 'setup':
        d.box(10,30,780,330,color=TEAL,fill='#051318')
        for i,color in enumerate(['#fb7185',GOLD,GREEN]):d.parts.append(f'<circle cx="{40+i*27}" cy="58" r="7" fill="{color}"/>')
        for i,(line,color) in enumerate([('$ nvidia-smi',MUTED),('$ nvcc --version',MUTED),('$ python3 tools/h100_preflight.py',TEAL),('H100 book preflight: PASS',GREEN)]):
            d.parts.append(f'<text x="45" y="{115+i*65}" font-family="DejaVu Sans Mono, monospace" font-size="26" fill="{color}">{escape(line)}</text>')
        d.text(400,410,'Connect → verify → compile → execute',25,MUTED,'middle')
    else:
        d.chip(310,60,'H100 / HOPPER',GREEN,250,235)
        for x,y,label,color in [(10,30,'HARDWARE',TEAL),(30,270,'CUDA / CuTe',GOLD),(610,150,'CUTLASS',TEAL)]:
            d.box(x,y,190,75,label,color,size=23)
            d.line(x+95,y+75,435,175,color,2)
        d.text(410,405,'Understand the machine. Verify the result.',24,MUTED,'middle')
    return ''.join(d.parts)


SPECS = [
    ('00-cover','FIELD GUIDE','Programming','the H100','CUDA, CuTe, and CUTLASS · from First Principles · Architecture, kernels, evidence','cover'),
    ('00-getting-started','GETTING STARTED','From an SSH shell','to a GPU result','The installation and first-run guide','setup'),
    ('01-architecture','CHAPTER 01','Inside the','Hopper machine','Put H100 in its architectural lineage','evolution'),
    ('02-clusters','CHAPTER 02','Blocks that','work together','Clusters · distributed shared memory · PTX','clusters'),
    ('03-barriers','CHAPTER 03','When is the','data ready?','Asynchrony needs an ownership protocol','barriers'),
    ('04-descriptors','CHAPTER 04','Describe a tile.','Choose its layout.','Tensor maps · coordinates · swizzling','descriptor'),
    ('05-tma','CHAPTER 05','Move bytes.','Keep computing.','Bulk copies and the Tensor Memory Accelerator','tma'),
    ('06-wgmma','CHAPTER 06','Meet the','Tensor Cores','Shared-memory operands · register accumulators','wgmma'),
    ('07-groups','CHAPTER 07','Queue the math.','Collect the result.','WGMMA groups · formats · complete epilogues','groups'),
    ('08-design','CHAPTER 08','Give each warp','a useful role.','Producer / consumer kernel design','specialization'),
    ('09-pipelines','CHAPTER 09','The art of','safe reuse','Stages · phases · persistent execution','ring'),
    ('10-streamk','CHAPTER 10','Distribute work,','not idle time.','Scheduling and split reductions','streamk'),
    ('11-multigpu','CHAPTER 11','One workload.','Many GPUs.','NVLink · NVSwitch · NCCL · parallelism','fabric'),
    ('12-attention','CHAPTER 12','Attention,','one tile at a time.','Online softmax meets Hopper scheduling','attention'),
    ('13-profiling','CHAPTER 13','Turn intuition','into evidence.','Profile the kernel. Test the explanation.','profile'),
]

for filename,chapter,line1,line2,subtitle,kind in SPECS:
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="520" viewBox="0 0 1440 520" role="img" aria-labelledby="title description">
<title id="title">{escape(chapter+': '+line1+' '+line2)}</title>
<desc id="description">{escape(subtitle)}. Original conceptual illustration, not a die photograph, measured trace or scale drawing.</desc>
<defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#071a20"/><stop offset="1" stop-color="#0e3033"/></linearGradient><radialGradient id="glow"><stop stop-color="#19685b" stop-opacity=".42"/><stop offset="1" stop-color="#071a20" stop-opacity="0"/></radialGradient><pattern id="dots" width="28" height="28" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#9ccfc0" opacity=".16"/></pattern><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto-start-reverse"><path d="M0 0 L8 4 L0 8" fill="none" stroke="#2dd4bf" stroke-width="1.5"/></marker></defs>
<rect width="1440" height="520" rx="20" fill="url(#bg)"/><rect width="1440" height="520" rx="20" fill="url(#dots)"/><ellipse cx="1040" cy="250" rx="610" ry="400" fill="url(#glow)"/>
<g font-family="Arial, Liberation Sans, sans-serif"><rect x="55" y="62" width="5" height="315" rx="2" fill="#a3e635"/>
<text x="83" y="94" font-size="19" letter-spacing="3" fill="#a3e635" font-weight="700">{escape(chapter)}</text>
<text x="83" y="176" font-size="45" fill="#eef9f5" font-weight="700">{escape(line1)}</text><text x="83" y="232" font-size="45" fill="#eef9f5" font-weight="700">{escape(line2)}</text>
<foreignObject x="83" y="265" width="400" height="110"><div xmlns="http://www.w3.org/1999/xhtml" style="font:22px/1.5 Arial,sans-serif;color:#a8c6c8">{escape(subtitle)}</div></foreignObject>
<text x="83" y="440" font-size="18" fill="#a8c6c8">PROGRAMMING THE H100</text>
<g transform="translate(555 30)">{illustration(kind)}</g><text x="1390" y="490" text-anchor="end" font-size="15" fill="#89b0b2">CONCEPTUAL ILLUSTRATION</text></g></svg>'''
    # Plain SVG text keeps the illustrations portable to static renderers.
    lines=subtitle.split(' · ')
    if len(lines)==1 and len(subtitle)>34:
        words=subtitle.split(); half=len(words)//2; lines=[' '.join(words[:half]),' '.join(words[half:])]
    replacement=''.join(f'<text x="83" y="{290+i*31}" font-size="21" fill="#a8c6c8">{escape(line)}</text>' for i,line in enumerate(lines))
    import re
    svg=re.sub(r'<foreignObject.*?</foreignObject>',replacement,svg,flags=re.S)
    (OUTPUT/(filename+'.svg')).write_text(svg+'\n')
print(f'Created {len(SPECS)} chapter and guide illustrations.')
