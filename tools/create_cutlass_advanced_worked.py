#!/usr/bin/env python3
"""Generate five numerical worked SVGs; no hardware results are generated."""
from pathlib import Path
from illustrations.drawing import Drawing, ROOT, INK, MUTED, LINE, COLORS

OUT = ROOT / 'docs/images/worked'


def canvas(chapter, slug, title, lede):
    d = Drawing(f'ch{chapter}/01-{slug}.svg', title, lede,
                'Derived worked example · numerical values and units are explicit · not a measured trace')
    d.path = f'worked/ch{chapter}-{slug}.svg'
    return d


def label(d, x, y, text, size=20, color=INK, center=False, weight=550):
    d.text(x, y, text, size, color, weight, 'middle' if center else 'start')


def node(d, x, y, w, h, heading, value, color='blue'):
    ink, tint = COLORS[color]
    d.rect(x, y, w, h, tint, LINE, 12)
    next_y = d.paragraph(x+16, y+30, heading.replace('→', '->'), w-32, 18, ink, 650, 24)
    d.paragraph(x+16, next_y+11, value.replace('→', '->'), w-32, 22, INK, 550, 29)


def stages():
    d = canvas(25, 'stage-generations', 'Three slots, eight tiles, distinct generations',
               'A cell identifies contents in one physical shared-memory slot. P/C are phase bits, not elapsed time.')
    y=d.y+22;left=146;cw=190
    for j in range(3):label(d,left+j*cw+cw/2,y,f'SLOT {j}',19,MUTED,True)
    y+=22
    for g in range(3):
        label(d,28,y+34,f'GEN {g}',20,MUTED)
        label(d,28,y+62,f'P={1^(g&1)} / C={g&1}',17,MUTED)
        for j in range(3):
            t=g*3+j;x=left+j*cw
            ink,tint=COLORS['blue' if j==0 else 'teal' if j==1 else 'violet']
            d.rect(x,y,cw-12,83,tint if t<8 else '#f4f5f7',LINE,10)
            label(d,x+18,y+36,f'tile {t}' if t<8 else 'NO TILE 8',23,ink if t<8 else MUTED)
            label(d,x+18,y+63,'32,768 raw bytes' if t<8 else 'do not wait on it',17,MUTED)
        y+=103
    d.y=y+4
    d.section('Slot 0 is the same storage on every reuse')
    y=d.y
    for i,(t,g) in enumerate([(0,0),(3,1),(6,2)]):
        x=28+i*242;node(d,x,y,220,94,f'tile {t} / generation {g}',f'P={1^(g&1)}, C={g&1}','blue')
        if i<2:d.line(x+222,y+48,x+238,y+48,MUTED,2,True)
    d.y=y+120
    d.note('Two different proofs', 'Full/ready: these bytes arrived. Safe release: every asynchronous reader finished. Only the second proof permits overwrite.', 'amber')
    d.note('Raw operand budget', 'A: 128 × 64 × 2 = 16,384 bytes. B: another 16,384 bytes. Three stages: 98,304 bytes before other storage.', 'teal')
    d.write()


def fusion():
    d=canvas(26,'fusion-values','Join numeric branches before applying ReLU',
             'Manual output coordinate (m,n)=(0,1): alpha=0.5, beta=0.25. All shown numbers are FP32 values.')
    y=d.y;node(d,28,y,324,108,'Complete MMA accumulator','17 × 0.5 = 8.5','blue')
    node(d,408,y,324,108,'C source at (0,1)','−4 × 0.25 = −1','violet')
    d.pathline([(190,y+108),(190,y+142),(380,y+142),(380,y+167)],MUTED,2,True)
    d.pathline([(570,y+108),(570,y+142),(380,y+142)],MUTED,2)
    node(d,180,y+174,400,104,'Join the two values','Z = 8.5 + (−1) = 7.5','teal')
    d.line(380,y+280,380,y+307,MUTED,2,True)
    node(d,180,y+314,400,104,'Activate, then store','ReLU(7.5) = 7.5 → D[0,1]','teal')
    d.y=y+448;d.section('A negative result exposes missing activation')
    node(d,28,d.y,704,118,'Illustrative changed C[0,0]=−2; accumulator=−1',
         '0.5 × (−1) + 0.25 × (−2) = −1 → ReLU → 0','amber')
    d.y+=142;d.note('Order is the operation', 'ReLU(alpha × acc + beta × C) is not alpha × ReLU(acc) + beta × C. Apply beta once after the full K reduction.')
    d.write()


def nibbles():
    d=canvas(27,'nibbles-scales-dot','Codes are not values; equal bytes need their scales',
             'Exact-scale CPU teaching model: M=1, K=16, N=2. The last ten K positions are zero; this is not native NVFP4 execution.')
    x0=108;cw=101;y=d.y+22
    label(d,28,y,'k',19,MUTED)
    for k in range(6):label(d,x0+(k+.5)*cw,y,str(k),19,MUTED,True)
    for name,values,col in [('value',[1,2,3,4,0,6],'blue'),('code',['2','4','5','6','0','7'],'violet')]:
        y+=22;label(d,28,y+35,name,19,MUTED)
        for k,v in enumerate(values):
            ink,tint=COLORS[col];d.rect(x0+k*cw,y,cw-6,58,tint,'#fff',7)
            label(d,x0+(k+.5)*cw-3,y+37,str(v),23,ink,True)
        y+=58
    y+=24
    for pair,(byte,hi,lo) in enumerate([('42','4','2'),('65','6','5'),('70','7','0')]):
        x=x0+pair*2*cw;mid=x+cw-3
        d.pathline([(x+cw/2,y-12),(x+cw/2,y+3),(mid,y+3),(mid,y+20)],MUTED,1.5,True)
        d.pathline([(x+1.5*cw,y-12),(x+1.5*cw,y+3),(mid,y+3)],MUTED,1.5)
        node(d,x,y+26,2*cw-8,130,f'byte {pair}\nhigh {hi} | low {lo}',f'0x{byte}','violet')
    d.y=y+185;d.section('B columns: same payload, different reconstruction')
    d.paragraph(28,d.y+4,'Both columns begin with packed bytes a1 c3 07; decoded values are 0.5, −1, 1.5, −2, 6, 0. Their normalized dot with A is −5.',704,20,INK,leading=28)
    d.y+=110;y=d.y
    node(d,28,y,342,130,'Column 0: sA=1, sB=2','1 × 2 × (−5) = −10','blue')
    node(d,390,y,342,130,'Column 1: sA=1, sB=8','1 × 8 × (−5) = −40','teal')
    d.y=y+156;d.note('Keep the axes and units', 'k is a logical position; a code is four bits; a packed byte holds two codes; a scale is a numeric factor. Swapping scales changes values without changing bytes.','amber')
    d.write()


def event_join():
    d=canvas(28,'event-join-boundaries','Join actual worker completion; label the timer scope',
             'Dependency diagram for the overlap program. Distances carry no duration; the lower strip is a separate hypothetical serial accounting example.')
    y=d.y;node(d,250,y,260,82,'Timing stream records','start','teal')
    d.pathline([(380,y+82),(380,y+106),(198,y+106),(198,y+132)],MUTED,2,True)
    d.pathline([(380,y+106),(562,y+106),(562,y+132)],MUTED,2,True)
    node(d,28,y+140,342,152,'Worker 0 waits on start','even chunks × 8\n64 transforms → done[0]','blue')
    node(d,390,y+140,342,152,'Worker 1 waits on start','odd chunks × 8\n64 transforms → done[1]','violet')
    d.pathline([(198,y+292),(198,y+323),(380,y+323),(380,y+346)],MUTED,2,True)
    d.pathline([(562,y+292),(562,y+323),(380,y+323)],MUTED,2)
    node(d,180,y+354,400,111,'Timing stream waits on BOTH','record stop; host synchronizes stop','teal')
    d.y=y+493;d.section('Paper accounting: 4 ms setup + ten serial requests')
    y=d.y+30;w=704;setup=w*4/6
    d.rect(28,y,setup,62,COLORS['amber'][1],'#fff',0)
    label(d,28+setup/2,y+39,'setup: 4 ms',22,COLORS['amber'][0],True)
    for i in range(10):d.rect(28+setup+i*(w-setup)/10,y,(w-setup)/10,62,COLORS['teal'][1],'#fff',0)
    label(d,28+setup+(w-setup)/2,y-12,'10 × 0.2 ms',20,COLORS['teal'][0],True)
    for xx,t in [(28,'0 ms'),(28+setup,'4 ms'),(732,'6 ms')]:label(d,xx,y+94,t,18,MUTED,xx!=28)
    d.y=y+122;d.note('Two valid scopes, different answers', 'Setup-inclusive: 6 ms / 10 = 0.6 ms per request. Prepared request: 0.2 ms. Neither is a newly measured H100 duration.','amber')
    d.write()


def corner():
    d=canvas(29,'boundary-tile','Prove the thin corner and its reduction tail',
             'M=130, N=70, K=35; tile 64 × 32 × 16. Counts and offsets below are not stored matrix values.')
    x0=208;cw=172;y=d.y+25
    for j,t in enumerate(['n=0..31','n=32..63','n=64..69']):label(d,x0+(j+.5)*cw,y,t,18,MUTED,True)
    y+=20
    counts=[['2048','2048','384'],['2048','2048','384'],['64','64','12']]
    for i,rr in enumerate(['m=0..63','m=64..127','m=128..129']):
        label(d,28,y+43,rr,18,MUTED)
        for j,v in enumerate(counts[i]):
            col='amber' if i==2 or j==2 else 'blue';ink,tint=COLORS[col]
            d.rect(x0+j*cw,y,cw-8,80,tint,'#fff',7)
            label(d,x0+(j+.5)*cw-4,y+35,v,25,ink,True)
            label(d,x0+(j+.5)*cw-4,y+61,'valid outputs',16,MUTED,True)
        y+=88
    d.y=y+18;d.section('Corner offsets: two rows × six columns')
    y=d.y+25;cw=80;x0=220
    for j in range(6):label(d,x0+(j+.5)*cw,y,str(64+j),18,MUTED,True)
    y+=18
    for i,m in enumerate([128,129]):
        label(d,28,y+37,f'row {m}',20,MUTED)
        for j,n in enumerate(range(64,70)):
            ink,tint=COLORS['teal' if m==129 and n==69 else 'violet'];d.rect(x0+j*cw,y,cw-5,59,tint,'#fff',6)
            label(d,x0+(j+.5)*cw-2,y+37,str(m*70+n),19,ink,True)
        y+=70
    d.y=y+14;d.note('Original bounds beat allocation bounds', 'Invalid (128,70) would produce offset 9030, aliasing valid (129,0). Predicate n < 70 before the access.','red')
    d.note('K tail and scale groups', 'K tile 2 names k=32..47: three real positions, thirteen neutral zeros. Blocks of 16 along K require ceil(35/16)=3 scale groups per A row or B column.','amber')
    d.write()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in [stages,fusion,nibbles,event_join,corner]:fn()
    print('Generated five numerical worked SVGs, width 760, embedded Inter and font license.')

if __name__ == '__main__':main()
