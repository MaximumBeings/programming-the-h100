"""Content-specific topology and dependency drawings; no imported diagram art."""
from drawing import COLORS,INK,MUTED,LINE

def reset(x):x.parts=list(x.header_parts);x.y=x.start_y

def node(x,xx,y,w,title,body='',color='teal',h=94):
    ink,tint=COLORS[color];x.rect(xx,y,w,h,tint,LINE,12)
    yy=x.paragraph(xx+w/2,y+30,title,w-24,20,ink,650,25,'middle')
    if body:x.paragraph(xx+w/2,yy+4,body,w-24,17,INK,450,23,'middle')

def chain(x,labels,colors=None):
    n=len(labels);gap=24;w=(704-gap*(n-1))/n;y=x.y
    for i,label in enumerate(labels):
        node(x,28+i*(w+gap),y,w,label[0],label[1] if len(label)>1 else '',colors[i] if colors else 'teal',112)
        if i<n-1:x.line(28+(i+1)*w+i*gap+3,y+56,28+(i+1)*(w+gap)-5,y+56,MUTED,2,True)
    x.y=y+138

def hardware(x):
    y=x.y
    x.rect(28,y,704,370,'#f8fafc');x.text(48,y+30,'GH100 COMPUTE ORGANISATION',15,MUTED,650)
    for r in range(2):
        for c in range(4):node(x,48+c*172,y+50+r*116,148,f'GPC {r*4+c}','SM groups','teal',94)
    node(x,48,y+282,664,'Shared L2 cache','50 MB on the H100 products described here','blue',82)
    for c in range(4):x.line(122+c*172,y+262,122+c*172,y+279,MUTED,2,True)
    y+=400
    node(x,28,y,220,'HBM memory','SXM: 80 GB HBM3','teal',100)
    node(x,270,y,220,'Peer GPUs','NVLink interfaces','violet',100)
    node(x,512,y,220,'Host platform','PCIe Gen5','blue',100)
    for xx in [138,380,622]:x.line(xx,y-34,xx,y-5,MUTED,2,True)
    x.y=y+128
    x.note('Physical layout versus enabled resources','The boxes show logical connections, not die placement. Full GH100 has 144 SMs; H100 SXM enables 132, and H100 PCIe enables 114.')

def sm(x):
    y=x.y
    x.rect(28,y,704,330,'#f8fafc');x.text(48,y+30,'ONE STREAMING MULTIPROCESSOR',15,MUTED,650)
    for i in range(4):node(x,48+(i%2)*342,y+52+(i//2)*120,322,f'SMSP {i}','Warp scheduling + register share','teal' if i%2==0 else 'blue',100)
    x.y=y+355
    chain(x,[('Shared / L1','Unified resource'),('Tensor Cores','Matrix work'),('TMA','Tensor movement')],['teal','violet','blue'])
    x.note('Resource totals','256 KiB of registers per SM; up to 228 KiB of shared memory. H100 provides 128 FP32 cores, 64 FP64 cores and four Tensor Cores per SM.')

def memorypath(x):
    chain(x,[('SM','Lane accesses'),('Cache path','L1 / L2 policy'),('HBM','External memory')],['violet','blue','teal'])
    x.note('Cache hits shorten the path','Transactions and cache policy determine how much traffic reaches HBM. Aggregate interface bandwidth is not a bandwidth allocation guaranteed to each SM.')

def gemm(x):
    y=x.y
    chain(x,[('HBM inputs','A and B tiles'),('TMA','Tensor request'),('Shared stage','Ready operands')],['teal','blue','teal'])
    y2=x.y+12
    node(x,270,y2,220,'WGMMA','Matrix operation','violet',105)
    x.pathline([(622,y+112),(622,y2+52),(495,y2+52)],MUTED,2,True)
    x.text(628,y+148,'consume',15,MUTED)
    y3=y2+148
    node(x,270,y3,220,'Registers','Distributed accumulator','violet',105)
    x.line(380,y2+108,380,y3-5,MUTED,2,True)
    node(x,28,y3,220,'Epilogue','Scale / convert / store','amber',105)
    x.line(267,y3+53,253,y3+53,MUTED,2,True)
    node(x,28,y3+148,220,'HBM output','Final C / D tile','teal',105)
    x.line(138,y3+108,138,y3+143,MUTED,2,True)
    x.y=y3+282
    x.note('Pipeline overlap','TMA may fill the next released stage while WGMMA reads the current ready stage. Wait for readers before reusing a stage, and for matrix completion before consuming results.')

def exchange(x):
    x.section('Global-buffer handoff')
    chain(x,[('Block A','Produce sum'),('Global buffer','Publish / wait'),('Block B','Read sum')],['teal','blue','teal'])
    x.section('Cluster shared-memory handoff')
    chain(x,[('Block A','Write own SMEM'),('DSMEM','Cluster protocol'),('Block B','Read A’s SMEM')],['teal','violet','teal'])
    x.note('The owner must stay alive','Complete remote accesses before the owning block exits. The DSMEM path avoids staging this partial sum through a global buffer; latency remains workload-dependent.')

def cluster(x):
    y=x.y
    for c in range(2):
        xx=28+c*362;x.rect(xx,y,342,270,COLORS['teal' if c==0 else 'blue'][1]);x.text(xx+20,y+33,f'Cluster {c}',23,COLORS['teal' if c==0 else 'blue'][0],650)
        node(x,xx+20,y+58,302,'Block / rank 0','Own shared-memory allocation','teal',80)
        node(x,xx+20,y+174,302,'Block / rank 1','Own shared-memory allocation','blue',80)
        x.line(xx+171,y+142,xx+171,y+169,MUTED,2,True)
        x.text(xx+184,y+160,'DSMEM',14,MUTED)
    x.y=y+299
    x.note('One GPC, separate clusters','Blocks in a cluster are co-scheduled within a GPC. Remote shared-memory access uses a rank and an offset within that cluster; readiness and lifetime remain explicit.')

def dsmem(x):
    y=x.y
    x.rect(28,y,704,350,'#f8fafc')
    x.text(48,y+32,'ONE THREAD BLOCK CLUSTER',16,MUTED,650)
    node(x,48,y+66,300,'Block A / rank 0','Own shared array: [1, 3]','teal',130)
    node(x,412,y+66,300,'Block B / rank 1','Own shared array: [2, 4]','blue',130)
    x.line(351,y+136,407,y+136,MUTED,2,True)
    x.line(409,y+163,353,y+163,MUTED,2,True)
    x.text(380,y+116,'DSMEM',14,MUTED,650,'middle')
    node(x,48,y+236,664,'Same offset; another rank','map_shared_rank(local_ptr, peer_rank)','violet',88)
    x.y=y+380
    x.note('Access does not transfer ownership','Each block keeps its own shared-memory allocation. Publish the data before a peer reads it, and keep the owning block alive until remote accesses finish.')

def multicast(x):
    y=x.y
    node(x,210,y,340,'Operand tile','One tensor request + destination mask','blue',106)
    x.line(380,y+108,380,y+142,MUTED,2)
    x.line(108,y+143,651,y+143,MUTED,2)
    for i in range(4):
        xx=28+i*181
        x.line(xx+80,y+143,xx+80,y+174,MUTED,2,True)
        node(x,xx,y+180,162,f'Rank {i}','Local SMEM','teal',105)
    x.y=y+314
    x.note('Four destinations, four local copies','Each selected block needs a matching destination layout and completion accounting. Multicast reduces redundant source requests; actual HBM bytes also depend on cache behaviour.')

def compileflow(x):
    chain(x,[('CUDA C++','.cu source'),('PTX','Virtual ISA'),('cubin / SASS','Target machine code')],['blue','violet','teal'])
    x.note('Compiler and loader roles','nvcc transforms CUDA device code; ptxas assembles the selected target. The driver loads supported code and may JIT compatible PTX when the binary supplies it.')
    x.note('Check architecture-specific targets','An sm_90a instruction path is not a general forward-compatibility guarantee for future GPUs. Use the compatibility rules for the actual binary and device.')

def tile(x):
    x.section('Global input: select a tile')
    x.matrix([['' for _ in range(8)] for _ in range(3)],[['teal' if r==1 and 2<=c<6 else 'gray' for c in range(8)] for r in range(3)],cell=70)
    y=x.y;x.line(380,y,380,y+30,MUTED,2,True);x.y=y+48
    chain(x,[('Shared tile','Cooperative load'),('Block sync','Publish readiness'),('Reuse','Compute from SMEM')],['teal','amber','violet'])
    x.note('Protect both handoffs','Finish the writes before reading another thread’s element. Finish the readers before overwriting the tile.')

def engines(x):
    y=x.y
    names=[('Warp schedulers','Issue eligible instructions','blue'),('Tensor Cores','Asynchronous matrix work','violet'),('TMA engine','Asynchronous tensor transfers','teal')]
    for i,(t,b,c) in enumerate(names):node(x,28+i*242,y,220,t,b,c,162)
    yy=y+224
    node(x,148,yy,270,'WGMMA completion','commit_group / wait_group','violet',115)
    node(x,458,yy,274,'TMA load completion','mbarrier accounting','teal',115)
    x.line(380,y+166,283,yy-5,MUTED,2,True);x.line(622,y+166,595,yy-5,MUTED,2,True)
    x.y=yy+145
    x.note('Three separate dependencies','Observe input readiness before reading operands, matrix completion before using results, and reader completion before reusing storage.')

def proxies(x):
    y=x.y
    node(x,28,y,285,'Generic accesses','Thread loads and stores','blue',110)
    node(x,447,y,285,'Async accesses','TMA / WGMMA operations','teal',110)
    yy=y+184
    node(x,210,yy,340,'Shared operand storage','One set of bytes; explicit handoffs','violet',110)
    x.line(171,y+114,290,yy-5,MUTED,2,True);x.line(589,y+114,470,yy-5,MUTED,2,True)
    x.text(380,y+145,'proxy ordering',18,MUTED,550,'middle')
    x.y=yy+140
    x.note('Order and complete the handoff','A proxy fence orders mechanisms; thread synchronisation coordinates participants; the applicable wait observes an operation’s completion.')

def handshake(x):
    y=x.y
    a=(28,y);b=(418,y);c=(418,y+215);d=(28,y+215)
    node(x,*a,314,'Producer','Wait EMPTY; issue transfer','blue',130)
    node(x,*b,314,'FULL','TMA data has arrived','teal',130)
    node(x,*c,314,'Consumer','Wait FULL; read operands','violet',130)
    node(x,*d,314,'EMPTY','Readers finished; stage released','amber',130)
    x.line(346,y+65,413,y+65,MUTED,2,True)
    x.line(575,y+134,575,y+210,MUTED,2,True)
    x.line(413,y+280,347,y+280,MUTED,2,True)
    x.line(185,y+210,185,y+135,MUTED,2,True)
    x.y=y+379
    x.note('Repeat for the next generation','The consumer observes the required reader completion before releasing the stage. Phase tracking distinguishes successive uses of the same physical barrier and storage.')

def mbarprotocol(x):
    y=x.y
    labels=['Producer thread','TMA hardware','Consumer threads']
    for i,label in enumerate(labels):
        xx=28+i*242;x.rect(xx,y,220,480,'#f8fafc',LINE,12);x.text(xx+110,y+32,label,18,MUTED,650,'middle')
    node(x,40,y+60,196,'Expect bytes','Issue the tile load','blue',110)
    node(x,282,y+188,196,'Transfer','Deliver operand bytes','teal',110)
    node(x,524,y+316,196,'Acquire ready','Read the current tile','violet',110)
    x.pathline([(236,y+115),(255,y+115),(255,y+243),(277,y+243)],MUTED,2,True)
    x.pathline([(478,y+243),(497,y+243),(497,y+371),(519,y+371)],MUTED,2,True)
    x.text(380,y+330,'FULL',16,COLORS['teal'][0],650,'middle')
    x.y=y+508
    x.note('Before the producer refills this stage','Finish all prior readers and signal the release / EMPTY handoff. Data-ready completion alone does not authorise overwriting an active reader’s storage.')

def im2col(x):
    x.section('Explicit materialisation')
    chain(x,[('Input image','Global memory'),('im2col','Global temporary'),('GEMM','Read temporary')],['blue','amber','violet'])
    x.section('Supported TMA im2col path')
    chain(x,[('Input image','Global memory'),('TMA im2col','Shared tile'),('WGMMA','Read shared tile')],['blue','teal','violet'])
    x.note('Avoid the intermediate global write/read','The tensor-map and window configuration must match a supported convolution layout. Input data still needs to be fetched.')

def ring(x):
    y=x.y
    pts=[(72,y),(448,y),(448,y+226),(72,y+226)]
    for i,(xx,yy) in enumerate(pts):node(x,xx,yy,240,f'Stage {i}',['FULL','LOADING','FULL','IN USE'][i],['teal','blue','teal','violet'][i],130)
    x.line(317,y+65,443,y+65,MUTED,2,True);x.line(568,y+135,568,y+221,MUTED,2,True)
    x.line(443,y+291,317,y+291,MUTED,2,True);x.line(192,y+221,192,y+135,MUTED,2,True)
    x.text(380,y+194,'N = 4',24,INK,650,'middle')
    x.y=y+389
    x.note('Advance index; prove ownership','stage_next = (stage + 1) mod 4. Arrows show index order. Readiness and release barriers decide when that next stage may actually be read or overwritten.')

def nvswitch(x):
    y=x.y
    for i in range(4):node(x,28+i*181,y,162,f'GPU {i}','','teal',70)
    yy=y+145
    for i in range(4):node(x,28+i*181,yy,162,f'Switch {i}','','blue',70)
    y2=yy+145
    for i in range(4):node(x,28+i*181,y2,162,f'GPU {i+4}','','teal',70)
    # Each GPU fans into the logical switch fabric; light lines remain legible.
    for i in range(4):
        for j in range(4):
            x.line(109+i*181,y+73,109+j*181,yy-3,'#b8c8d8',1)
            x.line(109+i*181,yy+73,109+j*181,y2-3,'#b8c8d8',1)
    x.y=y2+104
    x.note('Shared per-GPU budget','900 GB/s aggregate bidirectional NVLink = 450 GB/s each direction per GPU. Peer flows share that budget. The paths show logical DGX connectivity, not board traces.')

def rails(x):
    y=x.y
    x.text(28,y+20,'NODE A',18,MUTED,650);x.text(418,y+20,'NODE B',18,MUTED,650)
    for node_x in [28,418]:
        x.rect(node_x,y+42,314,154,'#f8fafc')
        node(x,node_x+15,y+59,132,'GPU 0','NIC 0','teal',112)
        node(x,node_x+167,y+59,132,'GPU 1','NIC 1','blue',112)
    sy=y+300
    node(x,28,sy,314,'Rail 0 fabric','Corresponding GPU / NIC position','teal',112)
    node(x,418,sy,314,'Rail 1 fabric','Corresponding GPU / NIC position','blue',112)
    x.pathline([(109,y+174),(109,y+230),(185,y+230),(185,sy-5)],COLORS['teal'][0],2,True)
    x.pathline([(499,y+174),(499,y+253),(215,y+253),(215,sy-5)],COLORS['teal'][0],2,True)
    x.pathline([(261,y+174),(261,y+216),(550,y+216),(550,sy-5)],COLORS['blue'][0],2,True)
    x.pathline([(651,y+174),(651,y+230),(605,y+230),(605,sy-5)],COLORS['blue'][0],2,True)
    x.y=sy+142
    x.note('Extend the pattern across nodes and positions','Each colour joins corresponding positions across nodes. Actual switch tiers and contention depend on the deployment. Supported SHARP operations can perform reduction work in the network.')

CUSTOM={
 'ch01/01':hardware,'ch01/02':sm,'ch01/12':memorypath,'ch01/24':gemm,
 'ch02/02':exchange,'ch02/03':cluster,'ch02/04':dsmem,'ch02/06':multicast,'ch02/07':compileflow,'ch02/09':tile,
 'ch03/04':engines,'ch03/05':proxies,'ch03/06':proxies,'ch03/10':mbarprotocol,'ch03/12':handshake,
 'ch05/05':im2col,'ch05/06':multicast,'ch08/07':ring,'ch09/01':handshake,'ch11/02':nvswitch,'ch11/03':rails}
def refine(x):
    key=x.path[:7]
    if key in CUSTOM:reset(x);CUSTOM[key](x)
