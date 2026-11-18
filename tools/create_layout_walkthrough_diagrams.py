#!/usr/bin/env python3
"""Generate original worked-coordinate SVGs, independent of chapter numbering.

Values are distinct coordinate-derived numbers; memory-strip labels are offsets.
All offsets use elements; FP32 conversion to bytes is shown separately.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
from illustrations.drawing import Drawing, ROOT, INK, MUTED, LINE, COLORS

OUTPUT = ROOT / 'docs/images/worked'
OUTPUT.mkdir(parents=True, exist_ok=True)


class WorkedDrawing(Drawing):
    def __init__(self, filename, title, lede):
        super().__init__('worked/' + filename, title, lede,
                         'Worked coordinate arithmetic · FP32 elements · No data movement implied by a view')
        # Use a chapter-independent header rather than Drawing's chapter parser.
        self.parts = []
        self.rect(0, 0, 760, 1600, '#ffffff', 'none', 0)
        self.text(28, 31, 'CUTE FOUNDATIONS  /  WORKED COORDINATE MAP', 14, MUTED, 650)
        self.line(28, 47, 732, 47, LINE, 1)
        y = self.paragraph(28, 85, title, 704, 29, INK, 650, 35)
        self.y = self.paragraph(28, y + 10, lede, 704, 19, MUTED, 450, 27) + 22

    def grid(self, title, values, selected, caption):
        self.section(title)
        rows, columns, cell = len(values), len(values[0]), 66
        left = (760 - columns * cell) / 2
        top = self.y + 26
        for column in range(columns):
            self.text(left + (column + .5) * cell, top - 11,
                      f'j={column}', 17, MUTED, 550, 'middle')
        for row in range(rows):
            self.text(left - 14, top + row * cell + 37,
                      f'i={row}', 17, MUTED, 550, 'end')
            for column in range(columns):
                chosen = (row, column) == selected
                fill = COLORS['teal'][0] if chosen else '#f2f6fa'
                color = '#ffffff' if chosen else INK
                self.rect(left + column * cell, top + row * cell,
                          cell - 3, cell - 3, fill, '#dce5ed', 8)
                self.text(left + (column + .5) * cell - 1,
                          top + row * cell + 39, values[row][column],
                          22, color, 650, 'middle')
        self.y = top + rows * cell + 18
        self.y = self.paragraph(28, self.y, caption, 704, 19, MUTED, 450, 27) + 22

    def strip(self, title, values, selected, columns=12):
        self.section(title)
        self.text(28, self.y, 'Element offsets above · Stored values inside', 17, MUTED, 450)
        self.y += 32
        cell = 704 / columns
        rows = (len(values) + columns - 1) // columns
        for offset, value in enumerate(values):
            row, column = divmod(offset, columns)
            x, y = 28 + column * cell, self.y + row * 92
            self.text(x + (cell - 3) / 2, y - 9, offset, 17, MUTED, 550, 'middle')
            chosen, hole = offset == selected, value is None
            fill = COLORS['teal'][0] if chosen else ('#eef0f3' if hole else '#edf3fc')
            color = '#ffffff' if chosen else (MUTED if hole else COLORS['blue'][0])
            self.rect(x, y, cell - 3, 50, fill, '#dce5ed', 7)
            self.text(x + (cell - 3) / 2, y + 33,
                      'pad' if hole else value, 19 if hole else 20,
                      color, 550 if hole else 650, 'middle')
        self.y += (rows - 1) * 92 + 74

    def calculations(self, items):
        self.section('Calculate the selected element')
        self.flow(items)


def row_column():
    d = WorkedDrawing('row-vs-column.svg', 'Same coordinate, different memory order',
                      'Logical shape M=4, N=3. Every value is 100i+j; highlight (2,1), whose value is 201.')
    values = [[100 * i + j for j in range(3)] for i in range(4)]
    d.grid('01  One logical matrix', values, (2, 1),
           'The grid shows values, not offsets. Moving between storage layouts preserves the logical value 201.')
    row = [values[i][j] for i in range(4) for j in range(3)]
    column = [values[i][j] for j in range(3) for i in range(4)]
    assert row[7] == column[6] == values[2][1] == 201
    assert sorted(row) == sorted(column)
    d.strip('02  Row-major storage · strides (3,1)', row, 7)
    d.strip('03  Column-major storage · strides (1,4)', column, 6)
    d.calculations([
        ('Choose the same logical element', '(i,j)=(2,1), so 100×2+1=201 in both layouts.', 'blue'),
        ('Row-major: count complete rows first', 'offset = 2×3+1×1 = 7 elements; FP32 displacement = 7×4 = 28 bytes.', 'teal'),
        ('Column-major: count complete columns first', 'offset = 2×1+1×4 = 6 elements; FP32 displacement = 6×4 = 24 bytes.', 'violet'),
    ])
    d.note('Check the convention', 'Row-major makes columns contiguous; column-major makes rows contiguous. The selected value is unchanged, while its storage offset differs.', 'amber')
    return d


def padding():
    d = WorkedDrawing('padding-and-storage.svg', 'Useful values and padding occupy different roles',
                      'Logical shape (3,4), value 100i+j. Highlight (2,3): the useful value is 203 in either allocation.')
    values = [[100 * i + j for j in range(4)] for i in range(3)]
    d.grid('01  One logical 3×4 matrix', values, (2, 3),
           'There are 12 useful coordinates. Padding slots are storage locations, not extra logical columns.')
    compact = [value for row in values for value in row]
    padded = [value for row in values for value in row + [None] * 4]
    assert compact[11] == padded[19] == values[2][3] == 203
    assert len(padded) == 24 and sum(value is None for value in padded) == 12
    d.strip('02  Compact rows · strides (4,1)', compact, 11)
    d.strip('03  Padded rows · strides (8,1)', padded, 19, columns=8)
    d.calculations([
        ('Compact placement', 'offset(2,3) = 2×4+3 = 11 elements; FP32 displacement = 44 bytes.', 'blue'),
        ('Padded placement', 'offset(2,3) = 2×8+3 = 19 elements; FP32 displacement = 76 bytes.', 'teal'),
        ('Size does not equal storage span', '12 useful elements; minimum padded span = 19+1 = 20. This strip reserves three full rows: 3×8 = 24 slots.', 'violet'),
    ])
    d.note('A view does not remove padding', 'Changing stride eight to four over the same pointer selects different values. A compact result requires a correctly indexed copy.', 'amber')
    return d


def transpose():
    d = WorkedDrawing('transpose-view.svg', 'Transpose the view, keep the same allocation',
                      'One six-element FP32 allocation. The same value 102 is original coordinate (1,2) and transposed coordinate (2,1).')
    original = [[0, 1, 2], [100, 101, 102]]
    transposed = [[original[i][j] for i in range(2)] for j in range(3)]
    assert original[1][2] == transposed[2][1] == 102
    assert all(transposed[j][i] == original[i][j] for i in range(2) for j in range(3))
    d.grid('01  Original view · shape (2,3), strides (3,1)', original, (1, 2),
           'Read each row across: 0,1,2 and 100,101,102. Coordinate (1,2) names value 102.')
    d.grid('02  Transposed view · shape (3,2), strides (1,3)', transposed, (2, 1),
           'Coordinates are exchanged. The transposed grid is a new interpretation, not a second packed allocation.')
    d.strip('03  The physical memory strip does not change',
            [value for row in original for value in row], 5, columns=6)
    d.calculations([
        ('Original coordinates', 'offset(1,2) = 1×3+2×1 = 5 elements.', 'blue'),
        ('Transposed coordinates', 'offset(2,1) = 2×1+1×3 = 5 elements. Both views select value 102.', 'teal'),
        ('Same physical word', 'The FP32 displacement is 5×4 = 20 bytes in both views. No bytes move merely because the view is constructed.', 'violet'),
    ])
    d.note('Keep coordinates, strides and lifetime together', 'Swapping only the shape is insufficient. Swap the corresponding stride modes too, and keep the underlying allocation alive.', 'amber')
    return d


def validate(path):
    root = ET.parse(path).getroot()
    assert root.get('viewBox', '').startswith('0 0 760 ')
    assert root.get('width') == '760'
    assert 'data:font/woff2;base64,' in path.read_text()
    texts = [element for element in root.iter() if element.tag.endswith('}text')]
    assert texts and min(float(element.get('font-size')) for element in texts) >= 14
    for element in texts:
        assert 0 <= float(element.get('x')) <= 760
        assert 0 <= float(element.get('y')) <= float(root.get('height'))


if __name__ == '__main__':
    assert 2 * 3 + 1 == 7 and 2 + 1 * 4 == 6
    assert 2 * 8 + 3 == 19 and 2 * 4 + 3 == 11
    assert 1 * 3 + 2 == 2 + 1 * 3 == 5
    for create in (row_column, padding, transpose):
        drawing = create()
        height = drawing.write()
        path = ROOT / 'docs/images' / drawing.path
        validate(path)
        print(f'WROTE {path.relative_to(ROOT)}: 760×{height}; embedded Inter, minimum font 14')
