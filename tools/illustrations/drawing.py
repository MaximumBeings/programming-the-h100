"""Original SVG primitives for the H100 technical illustration system."""
from pathlib import Path
from html import escape
import math,re
from PIL import ImageFont
ROOT=Path(__file__).resolve().parents[2]
FONT=ROOT/'tools/assets/Inter.ttf'
OLD=ROOT/'docs/images/chapter-openers/00-cover.svg'
old=OLD.read_text()
FONT_STYLE=(ROOT/'tools/illustrations/font-style.svg').read_text().strip() if (ROOT/'tools/illustrations/font-style.svg').exists() else re.search(r'<style type="text/css">@font-face.*?</style>',old,re.S)[0]
LICENSE=re.search(r'<metadata id="font-license">.*?</metadata>',old,re.S)[0]
COLORS={'teal':('#0f766e','#e7f4f0'),'blue':('#285a9f','#eaf0fc'),'amber':('#945b16','#fff2dd'),'red':('#a13e3e','#fbecec'),'gray':('#52657b','#f0f3f7'),'violet':('#6d55a0','#f1edfa')}
INK='#162b40';MUTED='#52657b';LINE='#d7e1e9'
METRICS={}
def width(s,size=20):
    if size not in METRICS:METRICS[size]=ImageFont.truetype(str(FONT),round(size*100),layout_engine=ImageFont.Layout.BASIC)
    return METRICS[size].getlength(s)/100 * 1.025

def wrap(s,w,size=20):
    output=[]
    for paragraph in str(s).split('\n'):
        line=''
        for word in paragraph.split():
            proposed=(line+' '+word).strip()
            if line and width(proposed,size)>w:output.append(line);line=word
            else:line=proposed
        output.append(line)
    return output

class Drawing:
    def __init__(self,path,title,lede,source='NVIDIA Hopper architecture / CUDA 12.8 PTX ISA'):
        self.path=path;self.title=title;self.lede=lede;self.source=source;self.parts=[];self.w=760
        self.rect(0,0,760,1600,'#ffffff','none',0)
        chapter=path.split('/')[0][2:];num=path.split('/')[1].split('-')[0]
        self.text(28,30,f'PROGRAMMING THE H100   /   CHAPTER {chapter}   /   FIGURE {num}',13,MUTED,650)
        self.line(28,46,732,46,LINE,1)
        y=self.paragraph(28,81,title,704,29,INK,650,35)
        self.y=self.paragraph(28,y+10,lede,704,18,MUTED,450,25)+24
        self.header_parts=list(self.parts);self.start_y=self.y
    def rect(self,x,y,w,h,fill='#fff',stroke=LINE,r=14):
        self.parts.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{r:g}" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
    def text(self,x,y,s,size=20,color=INK,weight=450,anchor='start'):
        self.parts.append(f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(s))}</text>')
    def paragraph(self,x,y,s,w,size=20,color=INK,weight=450,leading=None,anchor='start'):
        leading=leading or size*1.35
        for line in wrap(s,w,size):self.text(x,y,line,size,color,weight,anchor);y+=leading
        return y
    def line(self,x1,y1,x2,y2,color=MUTED,sw=2,arrow=False,dash=False):
        self.parts.append(f'<path d="M{x1:g},{y1:g} L{x2:g},{y2:g}" fill="none" stroke="{color}" stroke-width="{sw}"'+(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="5 5"' if dash else '')+'/>')
    def pathline(self,points,color=MUTED,sw=2,arrow=False):
        d='M'+' L'.join(f'{x:g},{y:g}' for x,y in points)
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
    def cardheight(self,title,body,w=336,size=20):
        return 48+len(wrap(title,w-54,22))*29+len(wrap(body,w-40,size))*27
    def card(self,x,y,w,title,body,color='teal',h=None,size=20):
        ink,tint=COLORS[color];h=h or self.cardheight(title,body,w,size)
        self.rect(x,y,w,h,tint,LINE)
        self.rect(x+18,y+22,4,22,ink,'none',2)
        yy=self.paragraph(x+34,y+39,title,w-54,22,ink,650,29)
        self.paragraph(x+20,yy+10,body,w-40,size,INK,450,27)
        return h
    def columns(self,items,n=2):
        gap=20;w=(704-gap*(n-1))/n;y=self.y
        for start in range(0,len(items),n):
            row=items[start:start+n];h=max(self.cardheight(t,b,w) for t,b,*_ in row)
            for i,item in enumerate(row):
                title,body,*col=item;self.card(28+i*(w+gap),y,w,title,body,col[0] if col else 'teal',h)
            y+=h+20
        self.y=y
    def flow(self,items):
        y=self.y
        for i,item in enumerate(items):
            t,b,*c=item;h=self.cardheight(t,b,632)
            self.rect(28,y+12,44,44,COLORS[c[0] if c else 'teal'][1],'none',22)
            self.text(50,y+41,str(i+1),20,COLORS[c[0] if c else 'teal'][0],650,'middle')
            self.card(100,y,632,t,b,c[0] if c else 'teal',h)
            y+=h
            if i<len(items)-1:self.line(416,y+2,416,y+24,MUTED,2,True)
            y+=32
        self.y=y-12
    def note(self,title,body,color='blue'):
        self.y+=4;self.y+=self.card(28,self.y,704,title,body,color)+20
    def section(self,title):
        self.text(28,self.y+22,title,23,INK,650);self.y+=48
    def table(self,headers,rows,widths=None):
        widths=widths or [704/len(headers)]*len(headers)
        y=self.y;self.rect(28,y,704,46,'#eaf0f5',LINE,9)
        x=28
        for h,w in zip(headers,widths):self.text(x+14,y+30,h,17,MUTED,650);x+=w
        y+=46
        for i,row in enumerate(rows):
            lines=[wrap(str(c),w-28,19) for c,w in zip(row,widths)]
            height=max(len(ls) for ls in lines)*26+24
            self.rect(28,y,704,height,'#fff' if i%2 else '#f8fafc',LINE,0)
            x=28
            for cell,w in zip(row,widths):self.paragraph(x+14,y+30,str(cell),w-28,19,leading=26);x+=w
            y+=height
        self.y=y+24
    def matrix(self,values,colors=None,caption=None,cell=58,rowlabels=None,collabels=None):
        rows=len(values);cols=len(values[0]);x=110 if rowlabels else (760-cols*cell)/2;y=self.y+32
        if collabels:
            for j,l in enumerate(collabels):self.text(x+(j+.5)*cell,y-13,l,16,MUTED,550,'middle')
        for i,row in enumerate(values):
            if rowlabels:self.text(x-16,y+(i+.5)*cell+6,rowlabels[i],17,MUTED,550,'end')
            for j,v in enumerate(row):
                color=colors[i][j] if colors else ('teal' if (i+j)%2==0 else 'blue')
                ink,tint=COLORS[color];self.rect(x+j*cell,y+i*cell,cell-3,cell-3,tint,'#fff',6)
                self.text(x+(j+.5)*cell-1,y+(i+.5)*cell+6,v,18,ink,650,'middle')
        self.y=y+rows*cell+22
        if caption:self.y=self.paragraph(28,self.y+5,caption,704,18,MUTED,leading=25)+15
    def timeline(self,lanes,steps=6):
        # [lane-name, [(start, duration, label, colour), ...]]
        left=150;cw=576/steps;y=self.y
        self.text(150,y+5,'Logical time',16,MUTED,450);self.line(265,y,726,y,MUTED,1.5,True);y+=43
        self.last_timeline=(left,cw,y)
        for i in range(steps+1):
            xx=left+i*cw;self.line(xx,y,xx,y+len(lanes)*90,'#e4ebf0',1);self.text(xx,y-8,str(i),14,MUTED,450,'middle')
        for name,events in lanes:
            self.paragraph(28,y+30,name,110,18,MUTED,650,24)
            for start,duration,label,color in events:
                x=left+start*cw;w=duration*cw-6;ink,tint=COLORS[color]
                self.rect(x,y+8,w,64,tint,'none',8)
                ls=wrap(label,w-14,17)
                yy=y+40-(len(ls)-1)*11
                for line in ls:self.text(x+w/2,yy,line,17,ink,550,'middle');yy+=22
            y+=90
        self.y=y+18
    def segments(self,title,segments,total=None):
        self.section(title);total=total or sum(v for _,v,_ in segments);y=self.y
        # Label values in an adjacent key; the bar shows proportions only.
        x=28
        for label,value,color in segments:
            ww=704*value/total;self.rect(x,y,ww,35,COLORS[color][0],'none',0);x+=ww
        self.y=y+62
        for label,value,color in segments:
            self.rect(28,self.y-15,14,14,COLORS[color][0],'none',3)
            self.text(54,self.y,label,19);self.y+=30
        self.y+=12
    def write(self):
        y=self.y+8
        self.line(28,y,732,y,LINE,1)
        y=self.paragraph(28,y+29,self.source,704,14,MUTED,450,20)+20
        height=math.ceil(y)
        # The background fits the actual illustration, rather than a fixed page.
        self.parts[0]=f'<rect width="760" height="{height}" fill="#fff"/>'
        svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}" viewBox="0 0 760 {height}" role="img" aria-labelledby="figure-title figure-desc">
<title id="figure-title">{escape(self.title)}</title><desc id="figure-desc">{escape(self.title+'. '+self.source+'. Original schematic illustration; not a physical die photograph or a measured performance trace.')}</desc>
<!-- original illustration system: h100-editorial-v1 -->
{LICENSE}{FONT_STYLE}<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="{MUTED}"/></marker></defs><g font-family="BookDiagram, Inter, Arial, sans-serif">{''.join(self.parts)}</g></svg>'''
        p=ROOT/'docs/images'/self.path;p.write_text(svg)
        return height
