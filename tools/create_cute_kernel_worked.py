#!/usr/bin/env python3
"""Numeric worked figures for the CuTe kernel track; no performance claims."""
from illustrations.drawing import Drawing, ROOT, INK, MUTED, LINE
import xml.etree.ElementTree as ET


class KernelDrawing(Drawing):
    def __init__(self, name, title, lede):
        super().__init__('worked/' + name, title, lede,
                         'Derived arithmetic from pinned CuTe v4.3.6 and the linked lab; not a measured GPU trace.')
        self.parts = []
        self.rect(0, 0, 760, 1600, '#fff', 'none', 0)
        self.text(28, 31, 'CUTE KERNELS  /  WORK THE NUMBERS', 14, MUTED, 650)
        self.line(28, 47, 732, 47, LINE, 1)
        y = self.paragraph(28, 85, title, 704, 29, INK, 650, 35)
        self.y = self.paragraph(28, y + 10, lede, 704, 19, MUTED, 450, 27) + 22


def copy():
    d = KernelDrawing('kernel-copy-tail.svg', 'Six owners, two padded address maps',
                      '37×70 FP32 copy: final tile origin (36,64). Offsets are elements from the guarded view origin.')
    d.section('1  Find the six valid logical coordinates')
    d.table(['Thread', 'Coordinate', 'Predicate'],
            [('0…5', '(36,64)…(36,69)', 'true'),
             ('6', '(36,70)', 'false: column'),
             ('32', '(37,64)', 'false: row')], [120, 328, 256])
    d.section('2  Apply each allocation’s own row pitch')
    d.table(['tid', 'Source: 36×73+n', 'Destination: 36×75+n'],
            [(str(t), str(36*73+64+t), str(36*75+64+t)) for t in range(6)], [90, 290, 324])
    d.note('One coordinate, one copied value',
           'tid 5: (36,69) to source[2697] to value 2590 to destination[2769]. FP32 byte offsets are 10788 and 11076.', 'teal')
    d.note('Padding is allocated, but not logical work',
           'tid 6’s (36,70) names padding at source 2698 / destination 2770. copy_if is false: no load, no store, no owner count.', 'amber')
    return d


def mma():
    d = KernelDrawing('kernel-mma-lane10.svg', 'Lane 10: register entries to output storage',
                      'SM75 m16n8k8 FP32 accumulator traits. The logical encoding is not the row-major store address.')
    d.note('Split the lane and value indices',
           'g=10/4=2; q=10%4=2. For value v: a=v%2, b=v/2. Encoded=32q+g+16a+8b.', 'blue')
    d.section('1  Decode the four accumulator entries')
    d.table(['Entry v', 'Encoded', '(row,column)'],
            [('0', '66', '(2,4)'), ('1', '82', '(2,5)'),
             ('2', '74', '(10,4)'), ('3', '90', '(10,5)')], [140, 220, 344])
    d.section('2  Store into a padded FP32 row-major tile')
    d.table(['Entry', 'row×ld+column; ld=12', 'Byte offset'],
            [('0', '2×12+4 = 28', '112'), ('1', '2×12+5 = 29', '116'),
             ('2', '10×12+4 = 124', '496'), ('3', '10×12+5 = 125', '500')], [140, 344, 220])
    d.note('Check the separation of contracts',
           'Traits::CLayout yields coordinates. The destination view yields element offsets. Register order 0,1,2,3 jumps eight rows after entry 1.', 'teal')
    return d


def banks():
    d = KernelDrawing('kernel-shared-banks.svg', 'One column, three bank mappings',
                      'Scalar FP32 word model: lane t reads logical (t,0). bank=word_offset modulo 32; byte_offset=4×word_offset.')
    d.section('1  Compare exact offsets before discussing conflicts')
    d.table(['Lane', 'Plain word/bank', 'Pad word/bank', 'XOR word/bank'],
            [(str(t), f'{32*t} / 0', f'{33*t} / {t}', f'{32*t+(0^t)} / {t}')
             for t in [0,1,2,3,7,15,31]], [80, 208, 208, 208])
    d.note('Plain: 32 different words, bank 0',
           '32t+0 modulo 32 is zero for every lane. This is not a same-word broadcast.', 'amber')
    d.note('Padding and XOR: 32 different banks',
           'Padding uses 33t. Word-model XOR uses 32t+(0 XOR t). Both visit banks 0…31 here, but their complete storage maps differ.', 'teal')
    d.section('2  Distinguish the maps away from column zero')
    d.table(['Coordinate', 'Plain', 'Padded', 'XOR'],
            [('(3,5)', '101; bank 5', '104; bank 8', '102; bank 6')], [160, 180, 180, 184])
    d.note('Instruction scope matters',
           'This is scalar FP32 arithmetic, not an FP16 or WGMMA transaction prediction. CuTe composition uses Swizzle<5,0,5> on word indices.', 'blue')
    return d


def gemm():
    d = KernelDrawing('kernel-gemm-ragged.svg', 'Follow one ragged output through K',
                      'Actual bounded input formulas: batch 1, output (36,28), M=37,N=29,K=41. Values below are hand-derived FP32 numbers.')
    d.note('Locate the owner and padded output',
           'block(x=1,y=2), thread(tx=12,ty=4) to (36,28). ldc=36; batch stride=1332. Output offset=1332+36×36+28=2656 elements.', 'blue')
    d.section('1  Keep the same accumulator across all tiles')
    d.table(['K origin', 'Real K coordinates', 'Partial dot', 'Running sum'],
            [('0', '0…15', '0.625', '0.625'),
             ('16', '16…31', '0.21875', '0.84375'),
             ('32', '32…40; 7 zero slots', '0.28125', '1.125')], [110, 250, 172, 172])
    d.section('2  Apply the epilogue once')
    d.table(['Branch', 'Substitution', 'Value'],
            [('Accumulator', 'alpha×sum = 0.5×1.125', '0.5625'),
             ('C source', 'beta×C = 0.25×0.5', '0.125'),
             ('Final output', '0.5625+0.125', 'D[2656]=0.6875')], [150, 320, 234])
    d.note('Load ownership differs from output ownership',
           'At K origin 32, this thread’s A load uses k=44 and is zero-filled. Its valid output still consumes k=32…40 from inputs supplied cooperatively.', 'amber')
    return d


def builder():
    d = KernelDrawing('kernel-builder-contract.svg', 'One B element, two coordinate conventions',
                      'Illustrative packed FP16 B: M=128,N=256,K=64,L=2. The adapter consumes a view; it does not transpose the allocation.')
    d.section('1  Match physical and CuTe coordinates')
    d.table(['Interface', 'Coordinate / strides', 'Element offset'],
            [('Mathematical B', '(l,k,n)=(1,7,11)', '16384+1792+11'),
             ('CuTe B modes', '(n,k,l)=(11,7,1)', '11+1792+16384'),
             ('Same element', 'StrideB=(1,256,16384)', '18187 = byte 36374')], [174, 300, 230])
    d.section('2  Advance only when the current check succeeds')
    d.table(['State', 'Actual adapter operation', 'Required check'],
            [('Arguments', 'shape, pointers, strides, scalars', 'allocation contract'),
             ('Accepted', 'gemm.can_implement(args)', 'CUTLASS success'),
             ('Prepared', 'workspace + gemm.initialize', 'CUTLASS success'),
             ('Launched', 'gemm.run()', 'CUTLASS success'),
             ('Completed', 'cudaDeviceSynchronize()', 'CUDA success'),
             ('Validated', 'CPU full-output oracle', 'zero errors')], [145, 330, 229])
    d.note('A successful launch is an intermediate state',
           'No earlier state substitutes for completion and numerical equality. The recorded harness uses its own three stated shapes; this rectangular coordinate trace is derived.', 'teal')
    return d


if __name__ == '__main__':
    for factory in [copy,mma,banks,gemm,builder]:
        d=factory();height=d.write();path=ROOT/'docs/images'/d.path
        tree=ET.parse(path).getroot();assert tree.get('width')=='760'
        assert 'data:font/woff2;base64,' in path.read_text()
        for e in tree.iter():
            if e.tag.endswith('}text'):
                assert float(e.get('font-size'))>=14
                assert 0<=float(e.get('y'))<=height
        print(path.relative_to(ROOT),height)
