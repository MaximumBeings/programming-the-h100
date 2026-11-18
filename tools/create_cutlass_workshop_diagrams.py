#!/usr/bin/env python3
"""Original, numerical SVG figures for the sequenced CuTe/CUTLASS workshop."""
from illustrations.drawing import Drawing, ROOT, INK, MUTED, LINE, COLORS

specs={
16:('Hierarchy is an address function','Keep logical size, storage span and nested coordinates separate.','nested'),
17:('A tensor is a view plus an engine','Tiling and slicing reorganize coordinates; a predicate protects the actual access.','tensor'),
18:('Trace the whole address chain','Example: M=4096, N=2048, K=1024; 128 × 128 output tiles; CTA (5,3).','cta'),
19:('Give every destination one owner','A 4 × 32 scalar copy tile uses 128 threads, one value per thread.','copy'),
20:('Which lane owns each output?','Warp MMA m16n8 FP32 accumulators: entries show lane:register, not memory addresses.','mma'),
21:('A permutation is not a bank proof','32-bit words, 32 banks, fixed column j=7: compare actual bank indices.','banks'),
22:('Build a complete reduction','A synchronous SIMT tile makes readiness, reuse and accumulator lifetime explicit.','gemm'),
23:('Assemble a checked Hopper GEMM','Pin the library; define the operation; check status, completion and every output.','builder'),
24:('Stages repeat; generations change','Three stages, eight K tiles: shared storage returns only after its readers finish.','pipeline'),
25:('Fuse the specified graph','Compute the complete linear combination before ReLU and the final FP32 store.','fusion'),
26:('A low-bit value needs its scale','Distinguish E2M1 storage and mathematical reconstruction from native Blackwell execution.','quant'),
27:('Measure the boundary you report','Kernel-only, request latency and concurrent throughput describe different intervals.','bench'),
28:('From a coordinate to evidence','Advance through the labs only when the previous contract is understood and checked.','workbook')}

for chapter,(title,lede,kind) in specs.items():
    (ROOT/f'docs/images/ch{chapter}').mkdir(exist_ok=True)
    d=Drawing(f'ch{chapter}/overview.svg',title,lede,'Original worked schematic · CuTe/CUTLASS workshop')
    d.parts=[part.replace('FIGURE overview.svg','WORKED MAP') for part in d.parts]
    if kind=='nested':
        d.table(['Quantity','Value','Meaning'],[
            ['Shape','((3,4),2)','24 logical coordinates'],
            ['Stride','((1,3),40)','offset = a + 3b + 40c'],
            ['Coordinate','((2,3),1)','2 + 9 + 40 = 51'],
            ['Storage span','52 elements','28 holes in this positive-stride view']], [180,170,354])
        d.columns([('First outer slice','Offsets 0–11; nested modes (3,4).','blue'),('Second outer slice','Offsets 40–51; the gap is not extra logical data.','violet')])
        d.note('A useful check','Enumerate a small domain and compare every offset before trusting a larger layout.')
    elif kind=='tensor':
        d.flow([('Engine + layout','A pointer-backed view names storage; an owning register tensor supplies storage.','blue'),
                ('Slice and tile','Select logical coordinates without copying their values.','violet'),
                ('Partition identity coordinates','Carry the original positions through the same thread/value mapping.','teal'),
                ('Predicate the operation','Only valid coordinates may load or store the allocation.','amber')])
    elif kind=='cta':
        d.table(['Step','Calculation','Element offset'],[
            ['Output origin','(5 × 128, 3 × 128)','(640,384)'],
            ['A row-major','640 × 1024 + 0','655360'],
            ['B column-major','0 + 384 × 1024','393216'],
            ['C row-major','640 × 2048 + 384','1311104'],
            ['Next K tile','k0 = 64','A +64, B +64']], [180,334,190])
        d.note('Bytes come last','Multiply the correct element offset by the element size, then add the allocation base.')
    elif kind=='copy':
        d.table(['Thread ID','Local (row,col)','Logical owner'],[
            ['0','(0,0)','First warp, first column'],['31','(0,31)','First warp, last column'],
            ['32','(1,0)','Second warp, first column'],['95','(2,31)','Third warp, last column'],
            ['127','(3,31)','Fourth warp, last column']], [150,190,364])
        d.note('Ragged 37 × 70 matrix','10 × 3 tiles assign 3840 positions: 2590 valid coordinates and 1250 false predicates.','amber')
    elif kind=='mma':
        values=[];colors=[]
        for m in range(16):
            row=[];cs=[]
            for n in range(8):
                lane=4*(m%8)+n//2;reg=2*(m//8)+n%2
                row.append(f'{lane}:{reg}');cs.append(['blue','teal','amber','violet'][(n//2)%4])
            values.append(row);colors.append(cs)
        d.matrix(values,colors,cell=58,rowlabels=[str(m) for m in range(16)],collabels=[str(n) for n in range(8)])
        d.note('128 outputs, 128 owners','32 lanes × 4 FP32 accumulators. This warp-level map is not the Hopper WGMMA fragment map.','amber')
    elif kind=='banks':
        d.table(['Row i','Plain / row-XOR','Column-XOR bank'],[[str(i),'7',str(7^i)] for i in range(8)],[170,267,267])
        d.columns([('Plain or row-XOR','All 32 lanes of the fixed-column access request bank 7.','red'),
                   ('Column-XOR','Across rows 0–31, j XOR i reaches all 32 banks.','teal')])
        d.note('Scope matters','This is a word-address teaching permutation, not a ready-made TMA 128-byte swizzle descriptor.')
    elif kind=='gemm':
        d.flow([('Cooperative loads','Each thread loads an A and B element, or writes zero for an invalid input.','blue'),
                ('Ready barrier','All block participants reach the rendezvous before reading shared operands.','teal'),
                ('Accumulate the K tile','Keep the output accumulator alive across the complete reduction.','violet'),
                ('Reuse barrier and epilogue','Finish shared readers before overwrite; apply alpha and beta once.','amber')])
    elif kind=='builder':
        d.flow([('Mathematical contract','Shapes, types, physical layouts, alpha and beta.','blue'),
                ('Compatible collectives','Build the epilogue, reserve its storage, select the mainloop.','violet'),
                ('Adapter lifecycle','Check arguments, workspace, initialization, launch and CUDA completion.','teal'),
                ('Independent reference','Check all batches, partial tiles and nonzero-beta output values.','amber')])
    elif kind=='pipeline':
        d.table(['K tile','Stage','Generation','P phase / C phase'],[[str(t),str(t%3),str(t//3),f'{1^((t//3)&1)} / {(t//3)&1}'] for t in range(8)],[140,140,180,244])
        d.note('Do not confuse the waits','Producer and consumer phases refer to their own empty/full barriers; the bit values need not match.','amber')
        d.note('Drain actual work','Never wait for a tile that was not submitted. Reuse follows completed asynchronous readers.')
    elif kind=='fusion':
        y=d.y
        d.card(28,y,342,'alpha × accumulator','FP32 compute for the stated finite-input contract.','blue',180)
        d.card(390,y,342,'beta × C','Use C once, after the full K reduction.','violet',180)
        d.pathline([(199,y+180),(199,y+212),(380,y+212),(380,y+246)],MUTED,2,True)
        d.pathline([(561,y+180),(561,y+212),(380,y+212)],MUTED,2)
        d.y=y+252
        d.flow([('Add the two terms','Z = alpha × AB + beta × C','teal'),('Activate, convert and store','D = ReLU(Z), not alpha × ReLU(AB) + beta × C.','amber')])
        d.note('Avoided intermediate pass','A separate FP32 Z write and read cost 8 × M × N × L bytes. Runtime benefit still requires measurement.')
    elif kind=='quant':
        d.table(['Positive E2M1 code','Magnitude','Interpretation'],[[str(i),str(v),'Sign bit supplies negative values'] for i,v in enumerate([0,.5,1,1.5,2,3,4,6])],[210,150,344])
        d.note('Representation contract','Packed nibbles, block scales, scale axes, zero blocks, rounding and any tensor scale must all agree.','violet')
        d.note('Architecture boundary','The CPU reference does not implement E4M3 scale rounding or native NVFP4. Hopper is not Blackwell.','amber')
    elif kind=='bench':
        d.flow([('Setup boundary','Allocation, compilation, packing and initialization: state whether included.','blue'),
                ('Kernel boundary','Measure the selected operation on its actual stream after warmup.','teal'),
                ('Useful-result boundary','Include dependencies, communication and conversion needed by the caller.','violet'),
                ('Evidence boundary','Keep raw samples, correctness checks, exact configuration and capture overhead separate.','amber')])
    else:
        d.flow([('Introductory: 15–18','Derive layouts, tensor views and complete coordinate/byte addresses.','blue'),
                ('Intermediate: 19–23','Prove ownership, inspect fragments, validate copy and GEMM implementations.','teal'),
                ('Advanced: 24–27','Explain pipeline generations, fusion semantics, precision and measurement.','violet'),
                ('Capstone: 28','Deliver checked sources, diagrams, actual execution records and a reproducible comparison.','amber')])
    d.write();print('Created chapter',chapter,kind)
