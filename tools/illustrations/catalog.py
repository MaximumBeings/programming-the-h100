#!/usr/bin/env python3
"""Original, content-specific technical illustrations for all twelve image chapters.

Run from the repository root with Pillow available. No GPU, browser or image
service is required. Chapter-opening illustrations are intentionally untouched.
"""
from pathlib import Path
import json,math
from drawing import Drawing,ROOT,COLORS,INK,MUTED,LINE
from specialized import refine
MADE=[]
SOURCES={
 'hardware':'Source: NVIDIA Hopper Architecture Whitepaper; H100 Tensor Core GPU datasheet.',
 'ptx':'Source: NVIDIA CUDA 12.8 PTX ISA 8.7; CUDA C++ Programming Guide.',
 'nccl':'Source: NVIDIA NCCL user guide; DGX H100 and SuperPOD reference architecture.',
 'attention':'Source: FlashAttention-3 paper (arXiv:2407.08608); NVIDIA PTX ISA.'}
def d(ch,n,title,lede,source=None):
    path=next((ROOT/'docs/images'/f'ch{ch:02}').glob(f'{n:02}-*.svg'))
    rel=str(path.relative_to(ROOT/'docs/images'))
    return Drawing(rel,title,lede,SOURCES[source or ('hardware' if ch==1 else 'nccl' if ch==11 else 'attention' if ch==12 else 'ptx')])
def end(x):
    refine(x)
    height=x.write();MADE.append({'file':'docs/images/'+x.path,'title':x.title,'lede':x.lede,'height':height,'width':760,'source':x.source})
def panels(ch,n,title,lede,items,ncols=2,note=None):
    x=d(ch,n,title,lede);x.columns(items,ncols)
    if note:x.note('What to remember',note)
    end(x)
def flow(ch,n,title,lede,items,note=None):
    x=d(ch,n,title,lede);x.flow(items)
    if note:x.note('What to remember',note)
    end(x)
def table(ch,n,title,lede,headers,rows,widths=None,note=None):
    x=d(ch,n,title,lede);x.table(headers,rows,widths)
    if note:x.note('What to remember',note)
    end(x)
def time(ch,n,title,lede,lanes,steps=6,note=None):
    x=d(ch,n,title,lede);x.timeline(lanes,steps)
    if (ch,n)==(10,5):
        left,cw,y=x.last_timeline
        for at,label in [(3,'launch hint'),(5,'A complete')]:
            xx=left+at*cw
            x.line(xx,y+74,xx,y+95,MUTED,2,True)
            x.text(xx+10,y+88,label,14,MUTED)
    if note:x.note('Reading this timeline',note)
    end(x)
def sparse(ch,n):
    x=d(ch,n,'Two values survive each quartet','2:4 structured sparsity stores values plus position metadata.')
    x.section('Logical dense row: two quartets')
    values=[['3.1','0','0','2.7'],['0','4.2','1.8','0']]
    colors=[['teal','gray','gray','teal'],['gray','blue','blue','gray']]
    x.matrix(values,colors,cell=100,rowlabels=['0–3','4–7'],collabels=['slot 0','slot 1','slot 2','slot 3'])
    x.columns([('Packed values','3.1, 2.7, 4.2, 1.8','teal'),('Position metadata','Quartet 0: positions 0 and 3\nQuartet 1: positions 1 and 2','blue')])
    x.note('Instruction-specific encoding','Metadata layout and participating lanes depend on the sparse instruction shape and precision. Check the matching PTX definition. Sparsity eligibility is a data-layout constraint, not a promise of a 2× application speedup.')
    end(x)
def swizzle(ch,n,bits=128):
    cols=bits//16;rows=cols
    x=d(ch,n,f'{bits}-byte swizzle: permuting 16-byte cells','A simplified aligned-row model shows how the row selects an XOR permutation.')
    values=[[str(c^(r&(cols-1))) for c in range(cols)] for r in range(rows)]
    colors=[['teal' if c==0 else 'gray' for c in range(cols)] for r in range(rows)]
    x.section('Columns: logical cells; entries: physical cells')
    x.matrix(values,colors,cell=min(68,560/cols),rowlabels=[f'row {r}' for r in range(rows)],collabels=[str(c) for c in range(cols)])
    x.note('Read a column, not just a row',f'physical_cell = logical_cell XOR (row & {cols-1})\nThe highlighted logical column visits different physical cells. Example: logical cell 0 in row 3 maps to cell 3 for the 128B model. The complete byte-address formula also depends on the aligned swizzle base.')
    end(x)

# CHAPTER 1 — Architecture. Original layouts emphasize scope and ownership.
x=d(1,1,'Inside the Hopper GPU','Logical GH100 organisation. Blocks are schematic, not physical die placement.')
x.section('Compute: eight graphics processing clusters')
x.matrix([[f'GPC {r*4+c}' for c in range(4)] for r in range(2)],cell=146)
x.note('Full die versus enabled product','Full GH100: 144 SMs. H100 SXM: 132 enabled SMs. H100 PCIe: 114 enabled SMs. Do not use one SKU’s SM count to predict another’s performance.')
x.columns([('Shared on-chip cache','50 MB L2 on the H100 products discussed here. It filters traffic between the SMs and external memory.','blue'),('External memory','H100 SXM: 80 GB HBM3, 3.35 TB/s product peak. PCIe memory bandwidth differs.','teal')])
x.columns([('NVLink fabric','Fourth-generation NVLink connects peer GPUs. State whether bandwidth is per direction or bidirectional.','violet'),('Host interface','PCIe Gen5 connects the GPU to the host platform. Host traffic has its own bandwidth budget.','gray')]);end(x)
x=d(1,2,'One SM, four processing partitions','Each SMSP schedules warps and uses a share of the SM’s execution resources.')
x.columns([(f'SMSP {i}','Warp scheduler\nRegister partition\nArithmetic and load/store issue','teal' if i%2==0 else 'blue') for i in range(4)])
x.note('Per-SM compute resources','H100: 128 FP32 cores, 64 FP64 cores and four fourth-generation Tensor Cores per SM. Register capacity totals 256 KiB per SM.')
x.note('Shared across the SM','The unified shared-memory / L1 resource is 256 KiB. Up to 228 KiB can be configured as shared memory; occupancy also constrains each block’s allocation.');end(x)
table(1,3,'Memory hierarchy: scope matters','Capacity, product peak bandwidth and a bank-throughput model are different quantities.', ['Level','Scope / capacity','Interpretation'],[
 ('Registers','Thread operands; 256 KiB per SM','Reuse operands without a memory load.'),('Shared memory','Up to 228 KiB per SM','32 four-byte banks; layout controls conflicts.'),('L2 cache','50 MB, GPU-wide','Hits and transaction efficiency depend on access patterns.'),('HBM3','80 GB, H100 SXM','3.35 TB/s product peak; sustained rates vary.')],[150,260,294],note='Shared-memory bank model at 1.98 GHz: 32 × 4 B × 1.98 GHz = 253.44 GB/s per SM. This is a model, not a measured kernel result.')
x=d(1,4,'Eight GPUs, one NVSwitch fabric','DGX H100 connects all eight GPUs through a switched NVLink network.')
x.matrix([[f'GPU {i}' for i in range(4)]],cell=150)
y=x.y;x.rect(100,y,560,70,'#eaf0fc');x.text(380,y+44,'NVSwitch fabric',25,'#285a9f',650,'middle')
for xx in [155,305,455,605]:x.line(xx,y-20,xx,y-3,MUTED,2,True);x.line(xx,y+73,xx,y+100,MUTED,2,True)
x.y=y+110;x.matrix([[f'GPU {i}' for i in range(4,8)]],cell=150)
x.note('Per-GPU bandwidth budget','900 GB/s bidirectional NVLink bandwidth = 450 GB/s in each direction. This budget is shared across peer traffic; it is not 900 GB/s per peer.');end(x)
time(1,5,'Serial work and pipelined overlap','Asynchronous copy is not unique to Hopper; TMA changes how bulk transfers are issued.',[
 ('Serial',[(0,2,'Load A','blue'),(2,1,'Math A','teal'),(3,2,'Load B','blue'),(5,1,'Math B','teal')]),
 ('Copy',[(0,2,'Load A','blue'),(2,2,'Load B','blue'),(4,2,'Load C','blue')]),
 ('Compute',[(2,2,'Math A','teal'),(4,2,'Math B','teal')])],note='Wait for a tile before reading it, and wait for its readers before reusing its buffer. Overlap is possible, not automatic.')
time(1,6,'A warp can work while data moves','Illustrative scheduling, not a measured utilisation trace.',[
 ('Transfer',[(0,2,'Tile 0','blue'),(2,2,'Tile 1','blue'),(4,2,'Tile 2','blue')]),('Consumer',[(0,2,'Prologue','gray'),(2,2,'Compute 0','teal'),(4,2,'Compute 1','teal')])],note='The steady state overlaps the next transfer with current-tile computation. The prologue and drain still cost time.')
panels(1,7,'From per-thread copies to a tensor request','The transfer request defines the division of work between threads and hardware.',[
 ('Per-thread copy','Threads calculate addresses and issue their pieces of the transfer. Ampere cp.async can already overlap copy and compute.','blue'),('Hopper TMA','A designated thread issues a descriptor-based tensor request. Hardware handles the requested addressing, transfer and supported layout operations.','teal')],note='A 64 × 64 BF16 tile contains 8 KiB of payload. Elements, warp instructions, memory transactions and sectors are not interchangeable counts.')
flow(1,8,'Two buffers, one repeating protocol','Never load a new tile over data that a consumer still needs.',[
 ('Prologue','Load tile 0 into slot A; establish its data-ready completion.','blue'),('First overlap','Read slot A while TMA fills slot B with tile 1.','teal'),('Rotate the slots','After readers release slot A, refill A with tile 2 while computing from B.','blue'),('Drain','Finish outstanding work before the buffers and barriers go out of scope.','gray')])
panels(1,9,'Where WGMMA operands live','Four warps cooperate on one warpgroup-level matrix operation.',[
 ('A operand','Depending on the instruction variant, A comes from shared memory or per-thread registers.','teal'),('B operand','B is supplied through a shared-memory matrix descriptor.','blue'),('C accumulator','Distributed accumulator registers provide the input accumulation state.','violet'),('D result','Result fragments remain distributed across the participating threads’ registers.','amber')],note='Use the exact shape/type layout in the PTX ISA. A register fragment is not an arbitrary contiguous matrix slice.')
sparse(1,10)
flow(1,11,'How a grid becomes resident work','A logical launch path; exact scheduler implementation is not an ISA contract.',[
 ('Launch metadata','Grid and block dimensions, register requirements, shared-memory allocation and cluster configuration.','blue'),('Find resident capacity','Respect warp, register, shared-memory and block limits. Clustered blocks must satisfy cluster placement requirements.','teal'),('Place a thread block','A block’s warps execute on an SM. Multiple eligible blocks may reside there together.','violet'),('Retire and replace','Completed blocks release resources so other work can become resident.','gray')])
flow(1,12,'The path from an SM to HBM','Cache hits can terminate a request before it reaches external memory.',[
 ('SM memory instruction','Lane addresses form memory transactions; coalescing affects the work the memory system sees.','teal'),('On-chip cache path','L1 behaviour depends on the operation and policy; L2 is shared across the GPU.','blue'),('Memory controllers','Misses that require external data are routed to HBM.','violet'),('HBM stacks','Product peak is an aggregate GPU-wide interface rate, not a per-SM guarantee.','amber')])
x=d(1,13,'A 128-byte cache line has four sectors','Sector granularity helps explain coalescing and overfetch.')
x.matrix([['0–31','32–63','64–95','96–127']],cell=160)
x.note('Count bytes and transactions separately','Each cell is a 32-byte sector. A request may touch only part of a line. Useful bytes, requested sectors, cache lines and HBM traffic need separate accounting.');end(x)
x=d(1,14,'A GPC contains multiple TPCs','Full-die schematic: nine texture processing clusters, two SMs per TPC.')
x.columns([(f'TPC {i}',f'SM {2*i}  +  SM {2*i+1}','teal' if i%2==0 else 'blue') for i in range(9)],3)
x.note('Cluster placement','Thread block clusters are scheduled within a GPC. Enabled product resources vary; this nine-TPC drawing describes the full GH100 organisation.');end(x)
panels(1,15,'A processing partition, closer up','SMSPs make warp scheduling and register pressure easier to reason about.',[
 ('Warp scheduling','Choose an eligible warp and issue its next instruction when dependencies and execution resources permit.','blue'),('Registers','Each partition has a register-file share. Per-thread register demand affects residency and spilling.','teal'),('Execution paths','FP32, FP64, integer, Tensor Core, load/store and special-function operations use different resources.','violet'),('Latency hiding','Another eligible warp can issue while one warp waits. Dependencies and low residency can reduce that choice.','amber')])
x=d(1,16,'A block is a collection of warps','256 threads make eight warps of 32 threads each.')
x.columns([(f'Warp {i}',f'Threads {32*i}–{32*i+31}','teal' if i%2==0 else 'blue') for i in range(8)],2)
x.note('Placement versus correctness','Warps of a block execute on the same SM and can use block shared memory. Correct code must not rely on an undocumented warp-to-partition assignment or fixed warp issue order.');end(x)
flow(1,17,'What makes a warp eligible to issue?','A conceptual dependency-to-execution loop, not a hardware pipeline timing diagram.',[
 ('Check dependencies','Are source operands ready? Is a prior operation or synchronisation still pending?','blue'),('Check resource availability','Can the required execution path accept this instruction?','violet'),('Select eligible work','The scheduler chooses among eligible resident warps.','teal'),('Track completion','Future dependent instructions must respect the issued operation’s completion.','gray')])
panels(1,18,'Issue bandwidth and execution bandwidth','An instruction must be issued before an execution unit can do useful work.',[
 ('Instruction issue','Scheduling and dispatch can limit the number of operations launched.','blue'),('Execution capacity','Arithmetic, Tensor Core and memory paths each have throughput and latency constraints.','teal'),('Dependency chains','A ready execution unit does not help if operands are not ready.','amber'),('Practical diagnosis','Use Nsight Compute to distinguish issue limits, dependencies and saturated execution resources.','violet')],note='Issue bandwidth, dependencies and execution demand jointly determine progress. Use measured stall and pipe metrics to identify the limiting resource.')
panels(1,19,'Coalescing: the same payload, different traffic','A warp’s lane addresses determine which sectors it touches.',[
 ('Contiguous lanes','32 float values at consecutive, properly aligned addresses span 128 bytes: four 32-byte sectors.','teal'),('Strided lanes','Addresses 128 bytes apart can touch 32 distinct sectors and cache lines for the same 128 useful bytes.','amber')],note='Cache policies and hits affect downstream traffic. Four useful sectors do not necessarily mean one physical HBM transaction.')
x=d(1,20,'Shared memory and L1 share a resource','H100 has a unified 256 KiB resource; supported carveouts are constrained by the product and kernel.')
x.segments('Maximum shared-memory configuration',[('Up to 228 KiB shared memory',228,'teal'),('Remainder of the unified resource',28,'blue')],256)
x.columns([('More shared memory','Can support larger tiles or more pipeline stages, but may reduce residency.','teal'),('More cache capacity','Can help cache-friendly traffic. It cannot replace an explicit shared-memory synchronisation protocol.','blue')]);end(x)
for ch,num in [(1,21),(2,13),(4,2)]:
 x=d(ch,num,'Shared-memory bank mapping','For ordinary four-byte words, addresses interleave across 32 banks.')
 x.matrix([[str(i*8+j) for j in range(8)] for i in range(4)],cell=70)
 x.note('Address-to-bank model','bank = floor(byte_address / 4) mod 32\nWord indices 0, 32 and 64 map to bank 0. Consecutive word indices 0–31 visit different banks.')
 x.columns([('Different banks','A suitable lane layout can service accesses without a bank conflict.','teal'),('Same bank, different words','Multiple addresses targeting one bank may require additional service. Same-address broadcast is a different case.','amber')]);end(x)
x=d(1,22,'A swizzle changes the physical bank pattern','A small XOR permutation makes column-wise access easier to see.')
x.section('Unswizzled: every row targets bank 0');x.matrix([['0','0','0','0']],colors=[['amber']*4],cell=130)
x.section('Swizzled example: rows spread across banks');x.matrix([['0','1','2','3']],cell=130)
x.note('Illustrative mapping','For a 32 × 32 float tile: physical_column = logical_column XOR (row mod 32). The addressing transformation must be applied consistently to both writes and reads.');end(x)
flow(1,23,'From thread block to SM execution','Resource requirements constrain residency; synchronisation controls correctness.',[
 ('Find room','The block needs warp slots, registers and shared memory.','blue'),('Place the block','All its warps reside on one SM. Cluster placement adds a group-level requirement.','teal'),('Execute eligible warps','Warps progress independently subject to dependencies and synchronisation.','violet'),('Synchronise shared work','Use the required block or cluster protocol before sharing or reusing data.','amber')])
flow(1,24,'A GEMM tile through the memory system','The main loop and epilogue have different dataflow and lifetime requirements.',[
 ('Describe the input tiles','Tensor maps specify supported tensor shapes, strides, tile extents and layout policy.','blue'),('Move global data to shared memory','TMA fills a stage; consumers observe data-ready completion.','teal'),('Issue matrix math','WGMMA reads its operands and accumulates distributed register fragments.','violet'),('Observe completion','Commit and wait as required before reading results or releasing consumed stages.','amber'),('Run the epilogue','Convert, scale and store the final output with the intended layout.','blue')],note='For different tiles, input transfer and matrix computation can overlap. This drawing does not assert a fixed number of cycles per stage.')

# CHAPTER 2 — Clusters, state spaces and representation.
panels(2,1,'A cluster adds a cooperation level','Grid → cluster → block → thread. A cluster groups blocks, not individual threads.',[
 ('Before clusters','Blocks can exchange global-memory data with a correct producer/consumer protocol. Block shared memory is not a cross-block address space.','blue'),('Hopper thread block cluster','Blocks in a cluster are co-scheduled within a GPC and can access distributed shared memory.','teal')],note='A portable cluster contains up to eight thread blocks. Larger configurations require device support and a nonportable launch configuration.')
flow(2,2,'Two ways to exchange a partial sum','The producer must finish writing before the consumer uses its result.',[
 ('Global-memory exchange','Block A writes a global buffer. After the required synchronisation and visibility, Block B loads it. Cache behaviour determines downstream HBM traffic.','blue'),('Cluster DSMEM exchange','Block A writes its shared memory. A cluster protocol makes the data ready, and Block B reads the remote shared-memory address.','teal')],note='DSMEM avoids staging this exchange through a global buffer. Preserve the owner block’s lifetime until every remote access has finished; there is no universal fixed cycle cost or speedup.')
x=d(2,3,'A cluster is a scheduling and lifetime contract','Co-scheduling enables cross-block shared-memory access within one GPC.')
x.columns([('Cluster 0','Block A / rank 0\nBlock B / rank 1','teal'),('Cluster 1','Block C / rank 0\nBlock D / rank 1','blue')])
x.section('Accessing another block’s shared-memory location')
x.flow([('Choose a rank','map_shared_rank(local_ptr, peer_rank) identifies the same offset in a peer block’s shared memory.','teal'),('Wait for readiness','All required writes must be visible before peers consume the data.','blue'),('Complete remote accesses','Do not let a block exit while another block may still access its shared memory.','amber')]);end(x)
x=d(2,4,'Distributed access, separate ownership','A cluster does not merge all shared-memory allocations into one flat physical SRAM.')
x.columns([('Block A / rank 0','Own shared array: [1, 3]\nMay access an allowed peer location.','teal'),('Block B / rank 1','Own shared array: [2, 4]\nRemains the owner of its allocation.','blue')])
x.note('Capacity model','Four blocks, each allocating S bytes, expose 4 × S bytes across the cluster. Each block still has its own allocation and must respect per-block and per-SM resource limits.');end(x)
table(2,5,'Compare exchange paths with evidence','Define the mechanism and measurement conditions before comparing runtimes.', ['Path','What the experiment must control'],[
 ('Global buffer','Bytes written and read, cache residency, synchronisation and memory ordering.'),('Cache-resident buffer','Warmup, working-set size, repeated access pattern and cache interference.'),('Cluster DSMEM','Cluster size, per-block allocation, remote access pattern and lifetime barriers.')],[190,514],note='Record transfer size, device SKU, clocks and timing scope so the measurements can be reproduced.')
x=d(2,6,'Multicast: one request, several destinations','A TMA multicast request delivers an operand tile to selected blocks in a cluster.')
x.note('Source request','One elected issuer requests a tile with a cluster destination mask.','blue')
x.matrix([[f'Rank {i}' for i in range(4)]],cell=150)
x.columns([('Local destinations','Each selected block receives the tile in its own shared-memory destination.','teal'),('Completion bookkeeping','Destination masks, shared-memory layout and barrier state must match the actual transfer.','blue')])
x.note('What is saved','Multicast can reduce redundant operand requests. A claim about HBM bytes must also account for caching and the actual memory-system traffic.');end(x)
flow(2,7,'CUDA source becomes GPU machine code','PTX is a virtual ISA. SASS is the machine-code instruction stream for a target architecture.',[
 ('CUDA C++ source','The compiler separates and transforms the relevant host and device code.','blue'),('PTX or device intermediate code','PTX describes typed virtual instructions and target requirements.','violet'),('Target-specific assembly','ptxas produces executable device machine code for a selected target.','teal'),('Load and launch','The driver loads supported code; a PTX JIT path is available only when the binary carries appropriate compatible PTX.','amber')],note='Architecture-specific targets such as sm_90a are not a blanket forward-compatibility promise. Match the compiled target and the instruction requirements to the executing GPU.')
x=d(2,8,'State spaces describe access and scope','Software state spaces do not map one-to-one to fixed cache capacities or universal latencies.')
x.columns([('Registers / .reg','Thread-private operands. H100 register storage totals 256 KiB per SM.','teal'),('Shared / .shared','Explicitly allocated block-local shared memory; cluster-qualified access can target peers.','blue'),('Global / .global','Device-visible allocation space, backed by the memory system and affected by caches.','violet'),('Local / .local','Per-thread address space, often used for spills; “local” does not mean on-chip SRAM.','amber'),('Constant / .const','Read-only constant address space with a specialised access path. The 64 KiB space is not a 64 KiB per-SM cache claim.','blue'),('Parameters / .param','Argument passing uses a defined parameter space; do not infer a fixed latency from the qualifier.','gray')]);end(x)
flow(2,9,'Load a tile, then reuse it','A 128-float tile contains 512 bytes; the benefit comes from controlled reuse.',[
 ('Cooperative global load','Threads load their assigned elements into the block’s shared-memory tile.','blue'),('Block readiness barrier','All participating threads finish the writes before other threads read them.','amber'),('Reuse in shared memory','Each thread can read the shared elements its calculation needs. Layout determines bank conflicts.','teal'),('Protect reuse','Complete the readers before overwriting or reusing the tile’s storage.','violet')])
table(2,10,'Floating-point formats: range and resolution','E is the exponent field; F is the stored fraction field. Subnormals and special values also matter.', ['Format','Fields','Finite range / role'],[
 ('FP32','1 S / 8 E / 23 F','Max ≈ 3.40 × 10^38; common accumulator format.'),('TF32','8 E / 10 F precision','Tensor Core computation mode with FP32-range exponent.'),('BF16','1 S / 8 E / 7 F','FP32-like range, fewer fraction bits.'),('FP16','1 S / 5 E / 10 F','Max 65,504; range must be checked.'),('FP8 E4M3','1 S / 4 E / 3 F','Max 448 for NVIDIA E4M3; no infinities.'),('FP8 E5M2','1 S / 5 E / 2 F','Max 57,344; supports infinity and NaN.')],[165,210,329],note='More exponent bits extend range. More fraction bits improve resolution. Wider range does not guarantee an overflow-free algorithm.')
x=d(2,11,'Two FP16 values in a 32-bit word','Packed operations interpret both halfwords; bit order must match the instruction.')
x.matrix([['hi: 31–16','lo: 15–0']],cell=285)
x.note('Packing convention','mov.b32 packed, {lo, hi}\nThe first 16-bit member occupies the low halfword. The second occupies the high halfword.')
x.columns([('Storage view','One 32-bit register carries two 16-bit encodings.','blue'),('Arithmetic view','Packed half-precision instructions can operate on the corresponding pair of values.','teal')]);end(x)
x=d(2,12,'GPU addresses count bytes','Typed C++ pointer arithmetic and raw byte offsets are different views of the same storage.')
x.table(['Value','Base address','Bytes occupied'],[('float a','0x1000','0x1000–0x1003'),('float b','0x1004','0x1004–0x1007'),('float c','0x1008','0x1008–0x100B')],[170,220,314])
x.note('The arithmetic','For float* p, p + 1 advances sizeof(float) = 4 bytes. In a raw PTX address expression, +4 is a literal four-byte offset.');end(x)
panels(2,14,'Generic pointers and address-space conversion','Treat generic address encoding as implementation-defined; do not invent fixed tag values.',[
 ('Generic pointer','A CUDA pointer can identify storage accessed through a supported generic address.','blue'),('State-space address','An instruction may require an address expressed for its particular state space.','teal')],note='Use supported conversions such as __cvta_generic_to_shared(). Never hard-code a high-bit pattern that supposedly means “shared” or “local”. UVA and state-space conversion are related address concepts, not interchangeable claims.')

# CHAPTER 3 — Dependency diagrams, not unsupported cache-coherency cartoons.
time(3,1,'Serial work leaves gaps','A deliberately serial dependency chain illustrates the cost of waiting.',[
 ('One task',[(0,2,'Fetch A','blue'),(2,1,'Use A','teal'),(3,2,'Fetch B','blue'),(5,1,'Use B','teal')])],note='Lengths are schematic. The ratio between a transfer and a computation depends on size, reuse, device state and the actual instruction stream.')
time(3,2,'Hide latency with independent work','The consumer may use a tile only after its required completion has been observed.',[
 ('Transfer',[(0,2,'Fetch A','blue'),(2,2,'Fetch B','blue'),(4,2,'Fetch C','blue')]),('Compute',[(2,2,'Use A','teal'),(4,2,'Use B','teal')])],note='Overlap changes the schedule, not the dependency. It does not guarantee that all transfer latency disappears.')
time(3,3,'Nonblocking issue separates request and use','The issuing thread can do other work before it needs the result.',[
 ('Blocking',[(0,1,'Issue','blue'),(1,4,'Wait','gray'),(5,1,'Use','teal')]),('Nonblocking',[(0,1,'Issue','blue'),(1,3,'Other work','teal'),(4,1,'Check','amber'),(5,1,'Use','teal')])],note='A nonblocking operation may still need a later wait. Polling or waiting too early can lose the intended overlap.')
x=d(3,4,'Three engines, different completion rules','Instruction issue, asynchronous matrix math and tensor movement can overlap.')
x.columns([('Warp schedulers','Issue eligible warp instructions.','blue'),('Tensor Cores','Execute matrix work launched by WGMMA.','violet'),('TMA engine','Transfer requested tensor data between supported memory spaces.','teal')],3)
x.columns([('WGMMA completion','commit_group → wait_group\nObserve completion before using the results.','violet'),('TMA load completion','mbarrier transaction tracking\nObserve readiness before reading shared-memory data.','teal')])
x.note('Coordinate each dependency','Observe input readiness before reading a tile, matrix completion before using results, and reader completion before reusing storage.');end(x)
panels(3,5,'Two proxies, one ordering problem','A proxy identifies the mechanism performing a memory operation.',[
 ('Generic proxy','Ordinary instruction-driven accesses, such as stores that initialise shared-memory operands.','blue'),('Async proxy','Operations such as TMA and asynchronous matrix instructions have their own ordering and completion requirements.','teal')],note='When the same bytes cross proxy boundaries, apply the required proxy fence and synchronisation. A proxy is a memory-model concept; it is not simply “the L1 cache” versus “the L2 cache”.')
flow(3,6,'Follow the operation, not a fixed cache cartoon','Different instructions can use different paths and ordering rules.',[
 ('Ordinary memory operation','A thread issues a supported load or store in the required address space.','blue'),('Async request','A thread launches a copy or matrix operation using its operands and descriptor state.','teal'),('Cross-proxy handoff','Use the prescribed ordering and readiness protocol before another mechanism consumes those bytes.','amber')],note='The memory model defines correctness. Cache behaviour varies with the instruction, address space and policy.')
flow(3,7,'RAW: consume only after production','A read-after-write dependency becomes a hazard when the consumer reads too early.',[
 ('Producer writes data','The source bytes must be fully prepared.','blue'),('Order the handoff','Use the proxy fence and inter-thread synchronisation required for this operation.','amber'),('Consumer reads valid data','The asynchronous operation may now consume the correctly published bytes.','teal')],note='A fence orders accesses; it does not replace every thread barrier or wait for every outstanding async operation.')
flow(3,8,'WAR: do not overwrite an active reader','The producer must not recycle a stage while its consumer still reads it.',[
 ('Consumer begins reading stage S','Its matrix operation may remain in flight after issue.','violet'),('Observe reader completion','Use the consumer’s applicable completion protocol.','amber'),('Release stage S','Signal that the storage may be reused.','teal'),('Producer overwrites stage S','A new transfer is safe only after the release is observed.','blue')])
panels(3,9,'An mbarrier tracks a phase of work','The object is opaque 64-bit shared-memory state; this is a semantic view, not a physical bitfield.',[
 ('Pending arrivals','Participants must satisfy the expected arrival count.','blue'),('Pending transactions','Expected async-transfer bytes must also be completed.','teal'),('Phase','Completion advances the phase so storage can be reused with a generation-aware wait.','violet'),('Completion condition','Both the required arrivals and transaction accounting must be satisfied.','amber')])
flow(3,10,'One tile through the mbarrier protocol','The exact instruction variant determines how the transfer is linked to completion.',[
 ('Initialise and publish','Create aligned barrier state and synchronise its users.','blue'),('Account for expected bytes','Set the correct transaction expectation for the transfer.','amber'),('Issue the async copy','Hardware delivers the tile while threads do independent work.','teal'),('Observe data readiness','The consumer uses the proper phase-aware acquire wait before reading.','violet'),('Release and reuse','Finish the readers before refilling the stage.','blue')])
time(3,11,'A buffer can be overwritten too soon','Two operations touching the same stage overlap unsafely in this example.',[
 ('Read S',[(0,4,'WGMMA reads S','violet')]),('Overwrite S',[(2,3,'TMA writes S','red')])],note='The overlap from logical time 2 to 4 is a lifetime violation. Data-ready completion for the new tile cannot make an unfinished old read safe.')
flow(3,12,'A full / empty handshake protects a stage','Data readiness and buffer reusability are separate facts.',[
 ('Producer waits for EMPTY','The previous readers have released the stage.','blue'),('Producer fills; FULL completes','The TMA transfer publishes the new tile’s readiness.','teal'),('Consumer waits for FULL','The consumer can read the current generation.','violet'),('Consumer finishes and signals EMPTY','Only then may the producer load another tile into that storage.','amber')])

# CHAPTER 4 — Tensor maps, bank permutations and boundary handling.
time(4,1,'Load the next tile while computing this one','Separate stages prevent the copy from overwriting the active operand.',[
 ('TMA',[(0,2,'Load 0 → A','blue'),(2,2,'Load 1 → B','blue'),(4,2,'Load 2 → A','blue')]),('WGMMA',[(2,2,'Read A','violet'),(4,2,'Read B','violet')])],note='Observe FULL before reading a stage and EMPTY before reusing it. The drawing is a schedule example, not a guaranteed throughput claim.')
swizzle(4,3,128)
x=d(4,4,'Narrower swizzles use smaller windows','The same XOR idea applies to fewer 16-byte cells per window.')
x.section('64 bytes: four cells');x.matrix([[str(c^r) for c in range(4)] for r in range(4)],cell=100)
x.section('32 bytes: two cells');x.matrix([[str(c^r) for c in range(2)] for r in range(2)],cell=110)
x.note('Choose a supported layout','Use the tensor-map and destination-alignment constraints for the chosen swizzle. A narrower tile does not automatically support every wider swizzle mode.');end(x)
panels(4,5,'L2 promotion is a fetch-policy choice','A promotion mode can request a larger cache granularity for tensor traffic.',[
 ('Smaller promotion','Can avoid bringing in adjacent data that will not be reused.','blue'),('Larger promotion','Can help when adjacent tensor accesses benefit from a wider cached region.','teal')],note='Policy is not a universal speedup. Measure the access pattern, reuse and actual traffic; do not assume every 32-byte request necessarily causes exactly 128 HBM bytes to be read and discarded.')
x=d(4,6,'A requested tile can cross the tensor edge','The descriptor defines the global tensor bounds; the requested box may extend past them.')
values=[['DATA' if r<4 and c<5 else 'FILL' for c in range(6)] for r in range(5)]
colors=[['teal' if r<4 and c<5 else 'amber' for c in range(6)] for r in range(5)]
x.matrix(values,colors,cell=83)
x.columns([('In bounds','Requested coordinates fetch valid source elements.','teal'),('Out of bounds','The selected supported fill mode supplies the specified values. Zero fill is not the same as every NaN-related fill mode.','amber')]);end(x)

# CHAPTER 5 — TMA operations and layout responsibilities.
flow(5,1,'After a tensor-copy request is issued','A high-level sequence; transfer granularity and timing are not fixed by this drawing.',[
 ('Issue the request','A thread supplies a descriptor, coordinates, destination and completion state.','blue'),('Generate and fetch','Hardware interprets the supported tensor layout and requests data.','teal'),('Deliver to shared memory','The transfer updates the appropriate completion accounting.','violet'),('Acquire and consume','The consumer observes readiness before using the tile.','amber')])
panels(5,2,'Per-thread copies versus bulk TMA','Both can be asynchronous. Their issue and addressing responsibilities differ.',[
 ('Ampere-style cp.async','Threads supply their own addresses and issue fixed-size copy operations.','blue'),('Hopper bulk / tensor request','A designated issuer supplies a bulk request or tensor coordinates; supported hardware handles the transfer.','teal')],note='Do not infer an instruction ratio from thread count multiplied by tile bytes. Compare equivalent payloads and count actual issued instructions for the implementation being tested.')
panels(5,3,'Tensor copy or raw byte copy?','The request format determines what the hardware knows about the data.',[
 ('Tensor-aware request','Tensor map + coordinates\nDescribes dimensions, byte strides, tile shape and supported transformations.','teal'),('Raw bulk request','Source / destination + byte count\nCopies a contiguous byte range without a multidimensional tensor map.','blue')])
x=d(5,4,'Request a rectangular tensor tile','For the example box, 32 columns × 16 rows = 512 elements.')
x.matrix([['' for c in range(8)] for r in range(6)], [['teal' if 2<=r<4 and 3<=c<7 else 'gray' for c in range(8)] for r in range(6)],cell=66)
x.note('Coordinate-based addressing','The tensor map supplies the global layout. Coordinates select the tile origin; box dimensions determine the requested extent. The highlighted rectangle is schematic, not an element-by-element scale drawing.');end(x)
panels(5,5,'im2col without a global intermediate','Compare dataflow, not an assumed format-independent instruction recipe.',[
 ('Explicit im2col','Image in global memory → materialised column matrix in global memory → GEMM.','blue'),('TMA im2col','Supported tensor-map/window configuration → column tile in shared memory → matrix computation.','teal')],note='The descriptor must use a supported tensor ordering and convolution-window configuration. The benefit is avoiding a separate global-memory materialisation step, not bypassing all input reads.')
x=d(5,6,'Multicast fans a tile out to a cluster','Each selected rank receives its own shared-memory copy.')
x.note('One tensor request','Operand tile + destination mask, for example ranks 0–3.','blue')
x.matrix([['Rank 0','Rank 1'],['Rank 2','Rank 3']],cell=220)
x.note('Local readiness still matters','Every receiving destination must have a valid layout and appropriate completion accounting. The mask describes destinations; it is not a promise of a fixed HBM-transaction count.');end(x)
x=d(5,7,'Align the tile, not just the allocation base','This layout example uses an eight-byte barrier and a 128-byte-aligned tile start.')
x.segments('Aligned dynamic shared-memory region',[('Barrier: bytes 0–7',8,'violet'),('Padding: bytes 8–127',120,'gray'),('Tile begins at byte 128',384,'teal')],512)
x.note('Round the offset upward','aligned_offset = (offset + alignment − 1) & ~(alignment − 1)\nFor offset = 8 and alignment = 128, the aligned offset is 128.')
x.note('Use the instruction’s actual constraint','Required alignment depends on the copy and swizzle mode. An invalid destination may fault or behave incorrectly; it is not guaranteed to fail silently.','amber');end(x)
time(5,8,'Prefetch creates an opportunity for reuse','The real load still needs its normal ordering and completion protocol.',[
 ('Prefetch',[(0,2,'Warm next tile','blue')]),('Compute',[(0,3,'Current tile','violet')]),('Real load',[(3,2,'Next tile','teal')])],note='Prefetch is advisory. Cache capacity, eviction and timing determine whether the real load benefits. No fixed microsecond latency is asserted.')

# CHAPTER 6 — Matrix collectives, descriptors and address/layout views.
table(6,1,'What changes with warpgroup MMA?','Compare the execution model while keeping precision and sparsity separate.', ['Aspect','Warp-level MMA','Warpgroup MMA'],[
 ('Participants','One warp','Four consecutive warps; 128 threads.'),('Completion','Warp-level instruction semantics','Async issue, commit and wait protocol.'),('Operand path','Register fragments','Shared-memory descriptors; supported A variants can use registers.'),('Data types','Includes FP16 / BF16 / TF32 and sparse variants on supported architectures','Adds supported Hopper FP8 paths; exact instruction variants matter.')],[150,250,304],note='Ampere already supports structured sparsity. Comparing A100 BF16 throughput directly with H100 FP8 mixes architectural and precision changes.')
x=d(6,2,'Four consecutive warps form a warpgroup','A WGMMA collective has 128 participating threads, not 128 independent matrix operations.')
x.matrix([['Warp 0','Warp 1','Warp 2','Warp 3']],cell=160)
x.note('One collective','All required participating warps execute the compatible aligned instruction sequence for the operation. Each thread owns only its fragment of the register result.')
x.matrix([['Warp 4','Warp 5','Warp 6','Warp 7']],cell=160)
x.note('Another warpgroup','A 256-thread block contains eight warps and can contain two warpgroups. They share the SM’s physical execution resources; this does not imply two dedicated Tensor Core units.');end(x)
time(6,3,'Overlap TMA with the WGMMA K-loop','The next stage may load while the current stage supplies matrix operands.',[
 ('TMA',[(0,2,'A0 / B0','blue'),(2,2,'A1 / B1','blue'),(4,2,'A2 / B2','blue')]),('WGMMA',[(2,2,'K tile 0','violet'),(4,2,'K tile 1','violet')])],note='Data-ready waits, proxy ordering, WGMMA completion and stage release are distinct responsibilities. A coloured overlap does not waive any of them.')
x=d(6,4,'ldmatrix: addresses and register fragments','For an x4 load, four 8 × 8 submatrices are supplied and distributed across a warp.')
x.section('Address-providing lanes for a 16 × 16 composition')
x.columns([('Top left','Submatrix 0\nAddress lanes 0–7','teal'),('Top right','Submatrix 2\nAddress lanes 16–23','blue'),('Bottom left','Submatrix 1\nAddress lanes 8–15','violet'),('Bottom right','Submatrix 3\nAddress lanes 24–31','amber')])
x.note('Providing an address is not owning a whole row','For an 8 × 8, 16-bit submatrix, four threads receive a row’s eight values: two values per thread. Address-provider and data-recipient roles are different.');end(x)
x=d(6,5,'Swizzled addresses combine row and column bits','For this aligned 128-byte-row, 16-bit-element example, row bits select a 16-byte-cell permutation.')
x.note('Illustrative byte-offset formula','linear = row × 128 + column × 2\nmask = ((linear >> 7) & 7) << 4\nswizzled_offset = linear XOR mask','teal')
x.table(['Row','Linear row start','XOR mask'],[(str(r),f'{128*r} bytes',f'{16*r} bytes') for r in range(4)],[150,300,254])
x.note('Keep the complete layout contract','Apply the required base alignment and use a matching descriptor. This small table does not claim every arbitrary stride is conflict-free.');end(x)
x=d(6,6,'The WGMMA shared-memory descriptor','A 64-bit descriptor encodes an address and traversal parameters; reserved bits remain zero.')
x.table(['Bit range','Field','Meaning'],[
 ('0–13','Address','Encoded shared-memory start address.'),('16–29','LBO','Leading-dimension byte offset encoding.'),('32–45','SBO','Stride-dimension byte offset encoding.'),('49–51','Base offset','Swizzled-layout matrix base offset.'),('62–63','Swizzle','0: none; 1: 128B; 2: 64B; 3: 32B.')],[150,170,384])
x.note('Encoding rule and reserved fields','Address, LBO and SBO use the specified 16-byte-granular encoding. Reserved bit ranges are 14–15, 30–31, 46–48 and 52–61. Use the exact matrix-layout definition, not a host pointer cast.');end(x)
x=d(6,7,'Descriptors describe jumps through a layout','LBO and SBO depend on the operand’s core-matrix ordering and swizzle mode.')
x.matrix([[f'{r},{c}' for c in range(4)] for r in range(3)],cell=120)
x.columns([('Leading offset','Describes the specified leading-dimension displacement in the PTX matrix layout.','teal'),('Stride offset','Describes the specified stride-dimension displacement in that layout.','blue')])
x.note('Coordinate sketch, not a universal formula','The cells illustrate repeated subblocks. “Next column” and “next row” are only meaningful after the instruction’s major mode and core-matrix layout have been selected.');end(x)

# CHAPTER 7 — Completion, matrix stores, low precision and sparsity.
time(7,1,'Commit a batch; wait for the required age','wait_group N allows up to N recent committed groups to remain pending.',[
 ('Issue',[(0,1,'G0','violet'),(2,1,'G1','violet')]),('GPU threads',[(1,1,'Prepare','blue'),(3,1,'Prepare','blue')]),('Observe',[(4,1,'Wait 1','amber'),(5,1,'Use G0','teal')])],note='After committing G0 and G1, wait_group 1 guarantees G0 has completed; G1 may still run. Reading G0 here requires registers disjoint from pending G1. Shared accumulators must wait for every group still using them.')
flow(7,2,'Ordering is different from completion','The correct sequence protects both inputs and outputs of asynchronous matrix math.',[
 ('Prepare operands','Apply the required register fence and shared-memory proxy ordering before WGMMA reads the operands.','blue'),('Issue and commit','Launch compatible WGMMA operations and form a committed group.','violet'),('Wait for the required completion','Observe the relevant groups before accessing accumulator results.','amber'),('Consume or store','Read, convert and store fragments only after their dependencies are satisfied.','teal')],note='wgmma.fence is not a wait for pending matrix arithmetic.')
x=d(7,3,'stmatrix: four address-providing quadrants','This is the same four-submatrix address ordering used for a 16 × 16 x4 composition.')
x.columns([('Top left','Rows 0–7, columns 0–7\nAddress lanes 0–7','teal'),('Top right','Rows 0–7, columns 8–15\nAddress lanes 16–23','blue'),('Bottom left','Rows 8–15, columns 0–7\nAddress lanes 8–15','violet'),('Bottom right','Rows 8–15, columns 8–15\nAddress lanes 24–31','amber')])
x.note('Collective means collective','All required warp lanes participate in the instruction. A selected lane’s pointer role does not make the instruction safe to issue from that lane alone.');end(x)
x=d(7,4,'stmatrix scatters register fragments','For each 8 × 8, 16-bit submatrix, four lanes contribute a row’s eight values.')
x.section('One 8 × 8 submatrix: lane ownership')
x.matrix([[str(4*r+c//2) for c in range(8)] for r in range(8)],cell=62)
x.note('Read one row','Row 0 is distributed over lanes 0–3, two adjacent values each. Row 1 uses lanes 4–7, and so on. Each x4 register corresponds to another submatrix.')
x.note('Do not assume every WGMMA layout matches automatically','The WGMMA accumulator shape, type conversion, register ordering and stmatrix variant must agree. A matching epilogue layout is a design requirement, not a universal free shuffle.');end(x)
x=d(7,5,'Two FP8 formats, different tradeoffs','The stored fraction bits affect resolution; exponent bits affect range.')
x.segments('E4M3 — eight bits',[('Sign: 1 bit',1,'gray'),('Exponent: 4 bits; bias 7',4,'blue'),('Fraction: 3 bits',3,'teal')])
x.segments('E5M2 — eight bits',[('Sign: 1 bit',1,'gray'),('Exponent: 5 bits; bias 15',5,'blue'),('Fraction: 2 bits',2,'teal')])
x.columns([('E4M3','Largest finite magnitude: 448. NVIDIA’s E4M3 encoding has NaN but no infinity.','teal'),('E5M2','Largest finite magnitude: 57,344. Supports infinity and NaN.','blue')])
x.note('Resolution is magnitude-dependent','FP8 does not distinguish every pair of nearby real numbers. Choose scaling and an error reference appropriate to the computation.');end(x)
x=d(7,6,'Pack FP8 values with an explicit bit order','A storage pack and an instruction operand pack are not automatically the same layout.')
x.section('Four values in a 32-bit word')
x.matrix([['31–24','23–16','15–8','7–0']],cell=160)
x.section('Two values in a 16-bit word')
x.matrix([['15–8','7–0']],cell=260)
x.note('Match the instruction','Use the packing and operand type required by the conversion, transport or MMA instruction. Do not infer a WGMMA register operand layout from the size of an arbitrary x4 storage type.');end(x)
sparse(7,7)

# CHAPTER 8 — Models and resource budgets.
x=d(8,1,'Two compute roofs, one memory ceiling','H100 SXM product-peak model. Arithmetic intensity uses useful FLOPs divided by transferred bytes.',source='hardware')
left,right,top,bottom=98,710,x.y+35,x.y+400
xx=lambda a:left+math.log10(a)/3*(right-left)
yy=lambda p:bottom-math.log10(p)/3*(bottom-top)
for v in [1,10,100,1000]:
 x.line(xx(v),top,xx(v),bottom,'#e3eaf0',1);x.text(xx(v),bottom+28,str(v),17,MUTED,450,'middle')
 x.line(left,yy(v),right,yy(v),'#e3eaf0',1);x.text(left-12,yy(v)+6,str(v),17,MUTED,450,'end')
x.text(left,top-17,'TFLOP/s',17,MUTED,550)
x.line(left,top,left,bottom,MUTED);x.line(left,bottom,right,bottom,MUTED)
for peak,color in [(989,COLORS['teal'][0]),(67,COLORS['blue'][0])]:
 points=[(xx(10**(i/80*3)),yy(min(peak,3.35*10**(i/80*3)))) for i in range(81)]
 x.pathline(points,color,3)
 ridge=peak/3.35
 x.line(xx(ridge),yy(peak),xx(ridge),bottom,color,1.2,dash=True)
 x.text(xx(ridge)+8,bottom-16,f'ridge {ridge:.0f}',15,color,550)
 x.rect(right-180,yy(peak)+6,176,26,'#fff','none',3)
 x.text(right-10,yy(peak)+25,'FP16 Tensor Core' if peak==989 else 'FP32 CUDA Core',17,color,650,'end')
x.y=bottom+69;x.text(404,bottom+60,'Arithmetic intensity: FLOP / byte',19,MUTED,450,'middle')
x.columns([('FP16 Tensor Core roof','Peak model: 989 TFLOP/s\nRidge: 989 / 3.35 ≈ 295 FLOP/byte','teal'),('FP32 CUDA Core roof','Peak model: 67 TFLOP/s\nRidge: 67 / 3.35 = 20 FLOP/byte','blue')]);x.note('Assumptions matter','Both lines use the 3.35 TB/s HBM product peak. Achieved throughput also depends on instruction efficiency, reuse, residency and other resource limits.');end(x)
time(8,2,'A producer can wait on a busy consumer','One qualitative signature of imbalance; use counters and timings to confirm it.',[
 ('Producer',[(0,1,'Load','blue'),(1,1,'Load','blue'),(2,4,'Wait for space','gray')]),('Consumer',[(1,2,'Math 0','violet'),(3,2,'Math 1','violet'),(5,1,'Math','violet')])],note='Producer waiting can mean the consumer is slower, but it does not by itself prove the kernel reaches a compute roof. Check dependencies, stage count and resource limits.')
flow(8,3,'Diagnose from evidence, then change one cause','Combine timing, traffic, instruction progress and resource constraints.',[
 ('Establish correctness and timing scope','Fix memory errors; distinguish kernel time, transfers and full-pipeline time.','blue'),('Inspect bytes and memory behaviour','Look at achieved bandwidth, cache transactions, coalescing and working-set reuse.','teal'),('Inspect instruction progress','Check issue eligibility, dependencies, execution-pipe demand and occupancy constraints.','violet'),('Form a specific hypothesis','Reduce bytes, increase reuse, break dependencies or rebalance producer / consumer work.','amber'),('Remeasure the same workload','Confirm correctness and compare the metric the change was intended to improve.','blue')])
panels(8,4,'Warp specialisation separates responsibilities','Dedicated producer and consumer roles can simplify a deep pipeline.',[
 ('Mixed-role warps','Each warp carries load state and computation state. Its own issue stream includes both kinds of work.','blue'),('Specialised warpgroups','A producer manages TMA and stage state; consumers perform matrix work on ready tiles.','teal')],note='Register allocation, stage depth and handoff cost determine whether this schedule helps. It is a resource tradeoff, not a guaranteed utilisation percentage.')
time(8,5,'Pipeline depth creates room for overlap','Deeper buffering helps only when its resource cost is affordable.',[
 ('Load',[(0,2,'Tile 0','blue'),(2,2,'Tile 1','blue'),(4,2,'Tile 2','blue')]),('Math',[(2,2,'Tile 0','violet'),(4,2,'Tile 1','violet')])],note='Two or more stages can hold ready, in-flight and consumed tiles. Extra stages consume shared memory and barrier state, and may lower occupancy.')
x=d(8,6,'Stage depth is a shared-memory budget','Example allocation: three stages, each with 32 KiB of A and 32 KiB of B.')
x.segments('Three operand stages',[('Stage 0: 64 KiB',64,'teal'),('Stage 1: 64 KiB',64,'blue'),('Stage 2: 64 KiB',64,'violet'),('Remaining budget: 36 KiB',36,'gray')],228)
x.note('Arithmetic','3 × (32 + 32) KiB = 192 KiB\n228 − 192 = 36 KiB before accounting for barriers, alignment and other live storage.')
x.note('A fourth stage does not fit this model','4 × 64 KiB = 256 KiB exceeds 228 KiB per SM. Per-block limits and residency constraints can be tighter than this simple capacity subtraction.','amber');end(x)
x=d(8,7,'A circular buffer recycles storage, not time','Producers and consumers track both stage ownership and generation.')
x.columns([('Stage 0','FULL: waiting to be consumed.','teal'),('Stage 1','LOADING: producer owns the destination.','blue'),('Stage 2','FULL: another ready tile.','teal'),('Stage 3','IN USE: consumer still owns the data.','violet')])
x.note('Advance only when the protocol permits','stage_next = (stage + 1) mod stage_count\nA wrap changes the generation / phase. The producer cannot assume it always stays a fixed number of stages ahead.');end(x)

# CHAPTER 9 — Lifetimes, generations and work ordering.
flow(9,1,'One stage, two ownership handoffs','FULL publishes data readiness; EMPTY publishes storage reusability.',[
 ('Producer owns an empty stage','Wait for EMPTY, then issue the transfer with the correct completion accounting.','blue'),('TMA completes the transfer','The FULL barrier becomes ready for the current generation.','teal'),('Consumer owns a full stage','Wait for FULL, then launch the operations that read its operands.','violet'),('Consumer releases the stage','Observe the readers’ completion before signalling EMPTY.','amber')])
x=d(9,2,'A stage index needs a generation','Four physical stages are reused repeatedly. Phase distinguishes successive passes.')
x.table(['Iteration','Stage','Phase'],[(str(i),str(i%4),str((i//4)%2)) for i in range(9)],[220,220,264])
x.note('Wrap the phase with the stage index','Iterations 0, 4 and 8 all name physical stage 0. Their expected phases are 0, 1 and 0. The phase-aware wait protocol must correspond to the particular generation in flight.');end(x)
x=d(9,3,'Reading can run ahead of release','Example: keep one newest WGMMA group pending while releasing an older stage.')
x.table(['Iteration','New read','Older release'],[('0','Stage 0','None yet'),('1','Stage 1','Stage 0, after its completion'),('2','Stage 2','Stage 1, after its completion'),('3','Stage 3','Stage 2, after its completion'),('4','Stage 0, new generation','Stage 3, after its completion')],[145,240,319])
x.note('The lag is a schedule choice','wait_group 1 can leave the most recent group pending while older groups complete. Group-to-stage bookkeeping must prove that the particular released stage is no longer being read.');end(x)
x=d(9,4,'Cooperation can split an output tile','Two consumer warpgroups share operand delivery while accumulating different output regions.')
x.columns([('Consumer 0','Rows 0–63 of a 128 × 128 output tile.\n8,192 FP32 outputs / 128 threads = 64 accumulator registers per thread.','teal'),('Consumer 1','Rows 64–127 of the same output tile.\nThe same 64-register accumulator count applies.','violet')])
x.note('Producer role','A producer warpgroup supplies the operand stages. Register demand also includes descriptors, temporaries and pipeline state.')
x.note('The cooperative gap','If both consumers enter an epilogue together, matrix work can pause. A staggered schedule may overlap one consumer’s epilogue with the other’s matrix work.');end(x)
time(9,5,'Stagger consumers to cover epilogue work','Illustrative ping-pong scheduling; the benefit depends on the relative task durations.',[
 ('Consumer 0',[(0,2,'Math 0','teal'),(2,1,'Epi 0','amber'),(3,2,'Math 2','teal'),(5,1,'Epi 2','amber')]),('Consumer 1',[(1,2,'Math 1','violet'),(3,1,'Epi 1','amber'),(4,2,'Math 3','violet')])],note='Ordered handoffs and buffer lifetimes must permit the overlap. The two consumers share physical Tensor Core capacity; staggering does not double that capacity.')
for ch,num,tiles in [(9,6,133),(10,1,140)]:
 x=d(ch,num,f'{tiles} tiles on 132 SMs: the last wave','Equal-duration tile model with one resident tile per SM. This is a scheduling illustration.')
 remain=tiles-132
 x.columns([('First wave','132 of 132 SMs can process one output tile.','teal'),('Second wave',f'{remain} of 132 SMs have a tile. Active-SM fraction: {remain}/132 = {remain/132:.1%}.','amber')])
 x.note('What persistent scheduling changes','A persistent CTA can take additional work without another launch. It does not make an indivisible final tile occupy every SM. Splitting the reduction work is a separate scheduling choice.');end(x)
x=d(9,7,'Tile order changes the reuse window','Numbers show visit order for a 4 × 4 output-tile grid.')
for title,values in [('Row-major',[[r*4+c for c in range(4)] for r in range(4)]),('Column-major',[[c*4+r for c in range(4)] for r in range(4)]),('Two-row grouped order',[[((r//2)*8+c*2+r%2) for c in range(4)] for r in range(4)])]:
 x.section(title);x.matrix(values,cell=75)
x.note('Locality is a working-set question','Changing visit order can keep useful A and B regions close in time. The benefit depends on tile shape, cache capacity, concurrent work and the actual operand traffic.');end(x)

# CHAPTER 10 — Split ownership and dependent launches.
x=d(10,2,'Partition a linear K-work space','Three output tiles × 90 K-iterations = 270 work units, divided among four CTAs.')
x.table(['CTA','Linear interval','Tile portions'],[
 ('0','[0, 67)','Tile 0: K [0, 67)'),('1','[67, 134)','Tile 0: [67, 90); tile 1: [0, 44)'),('2','[134, 202)','Tile 1: [44, 90); tile 2: [0, 22)'),('3','[202, 270)','Tile 2: K [22, 90)')],[100,180,424])
x.note('A split is a contribution, not an output tile','The intervals contain 67, 67, 68 and 68 iterations. A CTA can cross an output-tile boundary, so reduction and epilogue ownership need explicit rules.');end(x)
flow(10,3,'Three split roles, one final epilogue','A sequential workspace-reduction design for one output tile.',[
 ('First contribution','Compute the first K range and publish its partial accumulator plus progress.','blue'),('Middle contribution','Wait for the predecessor’s published state, combine the next partial result, then publish progress.','violet'),('Final contribution','Combine all required partial sums, then apply alpha, beta, bias and activation once.','teal')],note='The progress protocol needs the correct atomic memory ordering and initialisation. A plain counter increment is not automatically a publication fence.')
x=d(10,4,'Workspace separates partials from final output','Reduction metadata and numeric accumulators have different roles.')
x.columns([('Accumulator storage','A slot or region for the tile’s partial numeric state, with the required dtype and alignment.','blue'),('Progress metadata','Counters or flags describing which contribution is safely published.','violet')])
x.flow([('Initialise','Set progress and workspace according to the reduction protocol.','gray'),('Publish partial state','Write the partial accumulator before making it available to dependents.','blue'),('Acquire and reduce','A dependent contribution observes readiness before reading the numeric state.','teal'),('Finalise the output','Only the owner of the completed reduction applies the final epilogue.','amber')]);end(x)
time(10,5,'Dependent launch can overlap a tail','Programmatic dependent launch exposes a dependency point, not a real-time scheduling promise.',[
 ('Kernel A',[(0,3,'Main work','blue'),(3,2,'Tail','blue')]),('Kernel B',[(3,1,'Setup','teal'),(4,1,'Wait','gray'),(5,1,'Use A','violet')])],note='Kernel B may prepare dependency-independent state before its dependency wait completes. It must wait at the prescribed point before using Kernel A’s results. Scheduling benefit is workload-dependent.')
time(10,6,'Prefetch only dependency-independent inputs','A later kernel can prepare reusable weights while waiting on earlier output.',[
 ('Prefetch warp',[(0,1,'Setup','blue'),(1,2,'Warm weights','teal'),(3,2,'Wait','gray'),(5,1,'Load','blue')]),('Other warps',[(0,1,'Setup','blue'),(1,4,'Wait','gray'),(5,1,'Load','blue')])],note='The weight request must not depend on unfinished predecessor data. Cache residence and reuse determine whether prefetch reduces later traffic or latency.')
panels(10,7,'Align grouped work with operand reuse','Cluster peers can share a multicast operand only when their requests are compatible.',[
 ('Ungrouped work','Arbitrary K ranges can leave neighbouring CTAs asking for different operand slices.','amber'),('Grouped work','Compatible K offsets and tile coordinates can create shared operand requests for cluster multicast.','teal')],note='The exact grouping is a scheduler and kernel-layout decision. Confirm multicast eligibility, cluster launch constraints and measured reuse for that implementation.')

# CHAPTER 11 — Topology, collective semantics and parallel decomposition.
table(11,1,'A bandwidth hierarchy needs a direction','Representative DGX H100 / H100 SXM product-peak link budgets.', ['Interface','Budget','Scope'],[
 ('HBM3','3.35 TB/s','GPU-wide external-memory peak.'),('NVLink 4','450 GB/s each direction','900 GB/s bidirectional per GPU; shared across peer traffic.'),('PCIe Gen5 x16','About 64 GB/s each direction','Host interface signalling budget before protocol effects.'),('400 Gb/s network link','50 GB/s each direction','One nominal link; end-to-end rates include overhead and topology.')],[170,245,289],note='Bandwidth ratios compare like directions and scopes. Network latency, congestion, software protocol and message size also affect collective time.')
x=d(11,2,'NVSwitch gives peer GPUs a switched fabric','Logical DGX H100 connectivity; this drawing does not represent physical board routing.')
x.matrix([[f'GPU {i}' for i in range(4)]],cell=150)
x.note('Four third-generation NVSwitches','Every GPU connects into the fabric through its fourth-generation NVLink interfaces.','blue')
x.matrix([[f'GPU {i}' for i in range(4,8)]],cell=150)
x.note('Read the topology and the budget together','The switched fabric provides peer paths, but each GPU’s aggregate link budget is shared across traffic. A connection diagram does not grant the full budget independently to every simultaneous pair.');end(x)
x=d(11,3,'Rails organise the scale-out network','A rail groups corresponding GPU / NIC positions across nodes in a reference design.')
x.columns([('Rail 0','GPU / NIC position 0 across nodes uses the corresponding network plane.','teal'),('Rail 1','GPU / NIC position 1 uses another corresponding plane.','blue'),('Other rails','The same pattern extends across the remaining positions.','violet'),('SHARP-capable network','Supported reductions may aggregate data in network hardware.','amber')])
x.note('Reference architecture, not a universal deployment','Actual contention, switch tiers, rail selection and reduction offload depend on the installed topology and software configuration.');end(x)
x=d(11,4,'Six collective dataflow patterns','Use rank order and ownership to predict each output before running the collective.')
x.table(['Collective','Input → output relationship'],[
 ('Broadcast','One root’s buffer → the same buffer on every rank.'),('Reduce','All ranks’ values → a reduced result on one root.'),('AllReduce','All ranks’ values → a reduced result on every rank.'),('AllGather','One chunk per rank → rank-ordered concatenation on every rank.'),('ReduceScatter','Reduce full contributions → one reduced output chunk per rank.'),('All-to-All','Each source sends a distinct chunk to each destination.')],[210,494])
x.note('All ranks must agree','Use compatible participation, counts, datatypes and operations. A “root” is a rank in the communicator, not necessarily a physical CUDA device number.');end(x)
panels(11,5,'What gets split across GPUs?','Choose a decomposition that matches capacity and communication needs.',[
 ('Data parallelism','Replicate the model; process different data shards. Synchronise the required training state, commonly through gradient reduction.','teal'),('Tensor parallelism','Partition a layer’s tensor operations. Collective patterns depend on the row/column decomposition.','blue'),('Pipeline parallelism','Place successive model stages on different GPUs. Transfer activations and gradients between stages.','violet'),('Expert parallelism','Place experts across GPUs. Route token assignments and combine expert outputs, often using all-to-all exchange.','amber')])
x=d(11,6,'More microbatches amortise pipeline fill','Equal-duration GPipe stage model with N = 4 stages.')
for m in [4,8]:
 x.section(f'M = {m} microbatches')
 values=[['F' if s<=t<s+m else '·' for t in range(m+3)] for s in range(4)]
 colors=[['teal' if s<=t<s+m else 'gray' for t in range(m+3)] for s in range(4)]
 x.matrix(values,colors,cell=min(51,570/(m+3)),rowlabels=[f'Stage {s}' for s in range(4)])
 x.note('Idealised bubble fraction',f'(N − 1) / (M + N − 1) = 3 / {m+3} = {3/(m+3):.1%}. This panel shows the forward fill pattern; backward scheduling and activation memory require additional accounting.')
end(x)

# CHAPTER 12 — Attention dataflow, algebra and reported paper results.
x=d(12,1,'Attention intermediates dominate the simple I/O model','One head: N = 8,192, d = 128, BF16. Decimal MB and binary MiB are kept explicit.')
x.table(['Tensor or traffic','Bytes / size'],[
 ('Q, K, V and O together','4 × N × d × 2 = 8 MiB ≈ 8.39 MB.'),('One N × N score matrix','N × N × 2 = 128 MiB ≈ 134.22 MB.'),('Write and reread S and P','Four matrix-sized transfers = 512 MiB.'),('Naive minimum in this model','512 + 8 = 520 MiB ≈ 545.26 MB.')],[340,364])
x.note('What fusion changes','Tiled attention avoids materialising the full S and P intermediates in HBM. Actual fused I/O can include operand rereads; the 8 MiB useful-input/output size is not a measured total-traffic guarantee.');end(x)
# Compute the running statistics from the example, not rounded intermediate constants.
scores=[1.,3.,5.,2.];vals=[10.,20.,30.,40.];m=-math.inf;l=0.;o=0.;rows=[]
for i in range(2):
 ss=scores[2*i:2*i+2];vv=vals[2*i:2*i+2];new=max(m,max(ss));a=math.exp(m-new);p=[math.exp(v-new) for v in ss];l=a*l+sum(p);o=a*o+sum(pi*vi for pi,vi in zip(p,vv));m=new;rows.append((f'Block {i}',f'{m:.0f}',f'{l:.5f}',f'{o:.5f}'))
reference=sum(math.exp(s-max(scores))*v for s,v in zip(scores,vals))/sum(math.exp(s-max(scores)) for s in scores)
assert abs(o/l-reference)<1e-12
x=d(12,2,'Online softmax keeps sufficient statistics','Scores [1, 3, 5, 2], values [10, 20, 30, 40], processed in two blocks.')
x.table(['Step','m','l','Unnormalised O'],[('Initial','−∞','0','0')]+rows,[140,110,190,264])
x.note('When the maximum grows','alpha = exp(m_old − m_new)\nl_new = alpha × l_old + sum(exp(score − m_new))\nO_new = alpha × O_old + sum(exp(score − m_new) × value)','teal')
x.note('Final normalisation',f'O / l = {o/l:.5f}\nThe same result comes from a full stable-softmax reference. This scalar example demonstrates the recurrence, not an executed FA3 GPU kernel.');end(x)
x=d(12,3,'FA3 separates producers and consumers','A representative three-warpgroup organisation uses 384 threads.')
x.columns([('Producer / WG 0','Warps 0–3 prepare TMA transfers and pipeline state for operand tiles.','blue'),('Consumer / WG 1','Warps 4–7 compute attention for their assigned query rows.','teal'),('Consumer / WG 2','Warps 8–11 compute another query region with a staggered schedule.','violet')],3)
x.note('Shared circular buffer','K and V tile stages pass from producer ownership to consumer ownership and back through explicit readiness / release handoffs.')
x.note('Register budget','A producer can lower its register allocation while consumers request more, subject to the instruction’s allocation rules and the SM’s total resources. This is resource redistribution, not extra physical registers.');end(x)
time(12,4,'Stagger softmax and matrix work','A conceptual FA3 ping-pong schedule. Durations and utilisation are workload-dependent.',[
 ('Consumer A',[(0,2,'QK / PV','teal'),(2,1,'Softmax','amber'),(3,2,'QK / PV','teal'),(5,1,'Softmax','amber')]),('Consumer B',[(1,2,'QK / PV','violet'),(3,1,'Softmax','amber'),(4,2,'QK / PV','violet')])],note='Staggering can give Tensor Cores useful work while another consumer executes non-matrix operations. Ordered dependencies must still preserve the attention recurrence.')
time(12,5,'Pipeline matrix work within a consumer','Different attention blocks can occupy different dependency stages.',[
 ('QK',[(0,1,'QK 0','blue'),(2,1,'QK 1','blue'),(4,1,'QK 2','blue')]),('Softmax',[(1,1,'Block 0','amber'),(3,1,'Block 1','amber'),(5,1,'Block 2','amber')]),('PV',[(2,1,'PV 0','teal'),(4,1,'PV 1','teal')])],note='Maintain the running maximum, normaliser and output rescaling. Work can overlap only after the operands and statistics it depends on are valid; cell widths are illustrative.')
x=d(12,6,'Causal tiles fall into three categories','Rows are query blocks; columns are key blocks. Equal aligned block boundaries are assumed.')
values=[['FULL' if c<r else 'DIAG' if c==r else 'SKIP' for c in range(6)] for r in range(6)]
colors=[['teal' if c<r else 'amber' if c==r else 'gray' for c in range(6)] for r in range(6)]
x.matrix(values,colors,cell=84,rowlabels=[f'Q{r}' for r in range(6)],collabels=[f'K{c}' for c in range(6)])
x.columns([('FULL','All key positions are allowed by causality for this aligned tile.','teal'),('DIAG / SKIP','Diagonal tiles need an element mask; future-key tiles can be omitted.','amber')]);end(x)
x=d(12,7,'FA3 paper results and the ideas behind them','Reported author results on H100; these are not benchmark measurements from this book.')
x.columns([('FP16 forward','The paper reports up to 740 TFLOP/s and 1.5–2.0× speedup over its comparison baseline.','teal'),('FP8 path','The paper reports throughput close to 1.2 PFLOP/s, with a separate low-precision methodology.','violet')])
x.flow([('Warp specialisation','Overlap TMA data movement with matrix work.','blue'),('Interleave dependent stages','Schedule blockwise matrix products and softmax to reduce exposed gaps.','teal'),('Treat quantisation as its own algorithm','Block quantisation and incoherent processing address low-precision accuracy.','amber')])
x.note('Reproduction checklist','Match shape, head size, causal mode, dtype, scaling, hardware and timing scope. A reference softmax test does not reproduce the paper’s production-kernel performance.');end(x)

# Ensure every technical image has an original replacement, exactly once.
expected={str(p.relative_to(ROOT)) for p in (ROOT/'docs/images').glob('ch[0-9][0-9]/*.svg')}
actual=[item['file'] for item in MADE]
assert len(actual)==len(set(actual)), 'Duplicate illustration identifiers'
assert set(actual)==expected,(sorted(expected-set(actual)),sorted(set(actual)-expected))
(ROOT/'tools/illustrations/manifest.json').write_text(json.dumps(MADE,indent=2)+'\n')
print(f'Created {len(MADE)} original technical illustrations; chapter-opening artwork untouched.')
