#!/usr/bin/env python3
"""Original chapter maps for the Systems/CuTe/CUTLASS extension."""
from illustrations.drawing import Drawing, ROOT

specs={
14:('Systems before counters','Trace the application, investigate its dominant kernel, then verify the useful-result time.',[
 ('Application timeline','CPU submission, transfers, kernels, communication and waits.','blue'),
 ('Find the critical interval','Follow dependencies; overlapping activity is not additive wall time.','amber'),
 ('Inspect the selected kernel','Use Nsight Compute for its memory, instruction and resource limits.','teal'),
 ('Return to the application','Check results and compare unprofiled spans and a new Systems capture.','blue')]),
15:('Make the mapping explicit','A CuTe layout is an index function. A tensor view does not move the data.',[
 ('Logical coordinate','Shape defines the coordinate domain: for example (m, n).','blue'),
 ('Layout and composition','Strides and transforms map that coordinate to an element offset.','violet'),
 ('Thread and value ownership','Copy and MMA partitions assign coordinates to participants.','teal'),
 ('Physical operation','The pointer, layout and instruction contract determine the actual access.','amber')]),
16:('Assemble a checked Hopper GEMM','Pin the library revision; make layout, alignment and output checks explicit.',[
 ('Define the operation','D = alpha * A * B + beta * C; state shapes, types and physical strides.','blue'),
 ('Build compatible collectives','Select the epilogue, reserve its storage, then build the mainloop.','violet'),
 ('Initialize and launch','Check arguments, workspace, CUTLASS status and CUDA completion.','teal'),
 ('Verify every output','Nonuniform inputs, batches, edge tiles and an independent CPU oracle.','amber')]),
17:('Follow one tile through CUTLASS','A stage is reusable only after all asynchronous readers are finished.',[
 ('Acquire an empty stage','Use the right generation and a valid initialized pipeline.','blue'),
 ('TMA makes the stage full','The elected producer submits the intended bytes; consumers wait.','teal'),
 ('WGMMA reads the operands','Issue and complete arithmetic before releasing its shared inputs.','violet'),
 ('Fuse, store and measure','Preserve rounding and operation order; test the complete application.','amber')])}
for chapter,(title,lede,items) in specs.items():
 if chapter not in (14,15):continue
 (ROOT/f'docs/images/ch{chapter}').mkdir(exist_ok=True)
 d=Drawing(f'ch{chapter}/overview.svg',title,lede,'Original reading schematic · Chapters 14–17 extension')
 d.parts=[part.replace('FIGURE overview.svg','READING MAP') for part in d.parts]
 d.flow(items);d.write()
 print('Created chapter',chapter,'overview')
