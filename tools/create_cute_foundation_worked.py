#!/usr/bin/env python3
"""Original numeric inline figures for composition, tiling and CTA addresses."""
from create_layout_walkthrough_diagrams import WorkedDrawing, validate
from illustrations.drawing import ROOT, INK, MUTED, COLORS


class FoundationDrawing(WorkedDrawing):
    def strip_at(self, title, values, first_offset, selected):
        self.section(title)
        self.text(28, self.y, 'Element offsets above · Actual stored values inside', 17, MUTED)
        self.y += 31
        count = len(values)
        columns = min(count, 8)
        cell = 704 / columns
        rows = (count + columns - 1) // columns
        for index, value in enumerate(values):
            row, column = divmod(index, columns)
            offset = first_offset + index
            x, y = 28 + column * cell, self.y + row * 91
            chosen = offset in selected
            fill = COLORS['teal'][0] if chosen else '#eef3fa'
            color = '#fff' if chosen else INK
            self.text(x + (cell - 3) / 2, y - 9, offset, 17, MUTED, 550, 'middle')
            self.rect(x, y, cell - 3, 50, fill, '#dce5ed', 7)
            self.text(x + (cell - 3) / 2, y + 33, value, 20, color, 650, 'middle')
        self.y += (rows - 1) * 91 + 75


def composition():
    d = FoundationDrawing('composition-index-to-value.svg',
        'Composition: follow the intermediate index',
        'Outer A(i)=2i on 16 indices. Inner B(x)=3x on four indices. Storage word s contains value 100+s.')
    d.section('01  Apply B first, then A')
    d.table(['Logical x', 'B(x)', 'A(B(x))', 'Stored value'],
            [[str(x), str(3*x), str(6*x), str(100+6*x)] for x in range(4)],
            [150, 150, 190, 214])
    d.strip_at('02  A 32-word allocation · highlight selected offsets',
               list(range(100, 132)), 0, {0, 6, 12, 18})
    d.section('03  Reconstructed logical view')
    d.matrix([[100, 106, 112, 118]], [['teal']*4],
             collabels=['x=0', 'x=1', 'x=2', 'x=3'], cell=88,
             caption='These are stored values selected by the composition, not offsets. The intervening physical words still exist.')
    d.calculations([
        ('Choose x=3', 'B(3)=3×3=9. This is an index into A’s 16-index domain.', 'blue'),
        ('Apply the outer layout', 'A(9)=2×9=18 elements. With FP32 words, the displacement is 18×4=72 bytes.', 'teal'),
        ('Read only when an engine is used', 'A tensor using this layout would select stored value 118. Constructing the layout alone performs no read.', 'violet')])
    d.note('Check before reversing the order', 'CuTe accepts the reversed map as 16:6. For a finite four-index outer domain, A(2)=4 already exceeds its bounds.', 'amber')
    assert [2*(3*x) for x in range(4)] == [0, 6, 12, 18]
    return d


def predicated_tile():
    d = FoundationDrawing('tile-identity-and-predicate.svg',
        'A rounded tile is larger than its valid payload',
        'Row-major matrix (5,7), value 100i+j. Select tile (2,2) with shape (2,3): its origin is (4,6).')
    values = [[100*i+j for j in range(7)] for i in range(5)]
    d.grid('01  Original logical values · select (4,6)', values, (4, 6),
           'Highlight value 406. Logical bounds remain i<5 and j<7, regardless of the selected tile’s full extent.')
    d.section('02  Identity coordinates in the selected 2×3 tile')
    top = d.y + 29
    left, cell_w, cell_h = 116, 176, 79
    for i in range(2):
        for j in range(3):
            good = i == j == 0
            x, y = left+j*cell_w, top+i*cell_h
            fill = COLORS['teal'][0] if good else '#f0f2f5'
            color = '#fff' if good else MUTED
            d.rect(x, y, cell_w-5, cell_h-5, fill, '#dce5ed', 8)
            d.text(x+(cell_w-5)/2, y+29, f'({4+i},{6+j})', 21, color, 650, 'middle')
            d.text(x+(cell_w-5)/2, y+56, 'valid: value 406' if good else 'invalid: no access', 17, color, 450, 'middle')
    d.y = top + 2*cell_h + 25
    d.strip_at('03  Physical last-row excerpt · row stride seven',
               list(range(400, 407)), 28, {34})
    d.calculations([
        ('Recover global coordinates', 'Global = tile origin (4,6) + local coordinate. Local (0,0) is global (4,6).', 'blue'),
        ('Test original logical extents', '(4<5) and (6<7) is true. Every other coordinate in this tile fails at least one bound.', 'teal'),
        ('Separate coordinate, address and value', 'offset = 4×7+6 = 34 elements = 136 FP32 bytes; the stored value is 406.', 'violet')])
    d.note('The predicate protects the operation', 'The identity tensor supplies coordinates; copy_if suppresses invalid accesses. Creating a tile view alone does not supply bounds checks.', 'amber')
    assert [(i,j) for i in range(4,6) for j in range(6,9) if i<5 and j<7] == [(4,6)]
    return d


def cta_address():
    d = FoundationDrawing('cta-operands-to-result.svg',
        'One output: from CTA coordinates to a dot product',
        'M=5, N=7, K=6. A is row-major, B column-major, D row-major. A[m,k]=100m+k; B[k,n]=100k+n.')
    d.section('01  Choose the tile and one local point')
    d.table(['Quantity', 'Chosen value', 'Meaning'], [
        ['BM, BN, BK', '2, 3, 2', 'Output tile and K panel'],
        ['CTA (bm,bn), q', '(1,1), 2', 'Origins (m0,n0,k0)=(2,3,4)'],
        ['Local (mi,nj,kk)', '(1,2,1)', 'Global (m,n,k)=(3,5,5)']],
        [215, 180, 309])
    d.strip_at('02  A row m=3 · contiguous reduction values',
               list(range(300, 306)), 18, {23})
    d.strip_at('03  B column n=5 · contiguous column-major values',
               [100*k+5 for k in range(6)], 30, {35})
    d.section('04  Every K term contributes to D[3,5]')
    products = [(300+k)*(100*k+5) for k in range(6)]
    d.table(['k', 'A[3,k]', 'B[k,5]', 'Product'],
            [[str(k), str(300+k), str(100*k+5), str(products[k])] for k in range(6)],
            [100, 180, 180, 244])
    d.calculations([
        ('Locate the selected reduction operands', 'A[3,5]: 3×6+5=23 elements. B[5,5]: 5+5×6=35 elements.', 'blue'),
        ('Locate the output independently', 'D[3,5]: 3×7+5=26 elements. FP32 byte displacements: A 92, B 140, D 104.', 'teal'),
        ('Finish the complete reduction', f'At k=5 the contribution is 305×505={products[5]}. The sum over all six k values is {sum(products)}.', 'violet')])
    d.note('A K panel is not the entire answer', 'The selected q=2 panel supplies k=4 and k=5. Earlier panels must remain in the accumulator before D[3,5] is stored.', 'amber')
    assert sum(products) == 464575
    return d


if __name__ == '__main__':
    for create in (composition, predicated_tile, cta_address):
        drawing = create()
        height = drawing.write()
        path = ROOT / 'docs/images' / drawing.path
        validate(path)
        print(f'WROTE {path.relative_to(ROOT)}: 760×{height}; numeric checks, XML and embedded Inter validated')
