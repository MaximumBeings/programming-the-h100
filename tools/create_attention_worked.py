#!/usr/bin/env python3
"""Original numerical attention diagrams; no execution or performance claims."""
import math
from illustrations.drawing import Drawing, INK, MUTED


def canvas(chapter, name, title, scope):
    d = Drawing(f'ch{chapter}/01-{name}.svg', title, scope,
                'Derived exact-real attention arithmetic; not a GPU execution or timing result.' if chapter == 30 else
                'Pinned CUTLASS example 88 ownership and byte arithmetic; not a measured timeline.')
    d.path = f'worked/ch{chapter}-{name}.svg'
    return d


def softmax():
    d = canvas(30, 'online-softmax', 'Two GEMMs, one streamed softmax state',
               'Exact-real teaching case: two query rows, three keys, dimension d=2. Scale scores by 1/sqrt(2).')
    d.section('1  QK^T produces scores, not probabilities')
    d.table(['Input', 'Rows / meaning'], [
        ('Q', '[sqrt(2) ln(2), 0]; [0, sqrt(2) ln(2)]'),
        ('K', '[1,0]; [0,1]; [1,1]'),
        ('V', '[1,0]; [0,2]; [3,4]')], [100, 604])
    d.table(['Query', 'Scaled scores', 'exp(scores)', 'Sum'], [
        ('0', '[ln(2), 0, ln(2)]', '[2,1,2]', '5'),
        ('1', '[0, ln(2), ln(2)]', '[1,2,2]', '5')], [85, 260, 239, 120])
    d.note('The second GEMM multiplies normalized weights by V',
           'Row 0: (2[1,0] + 1[0,2] + 2[3,4])/5 = [8,10]/5 = [1.6,2]. Row 1 gives [7,12]/5 = [1.4,2.4].', 'teal')
    d.section('2  Stream row 0 without retaining the full score row')
    d.paragraph(28, d.y, 'State m is the largest score seen; l is a sum of shifted exponentials; z is a two-value shifted weighted sum. Final output is z/l.', 704, 20, INK, leading=28)
    d.y += 108
    d.table(['Keys processed', 'm', 'l', 'z'], [
        ('0 and 1', 'ln(2)', '1 + 0.5 = 1.5', '[1,1]'),
        ('Then key 2', 'ln(2)', '1.5 + 1 = 2.5', '[1,1] + [3,4] = [4,5]'),
        ('Normalize once', 'state complete', 'divide by 2.5', '[1.6,2]')], [170, 140, 160, 234])
    d.section('3  A larger maximum must rescale the old state')
    d.table(['Alternative order', 'Rescaling and result'], [
        ('Key 1 first', 'score 0: m=0, l=1, z=[0,2]'),
        ('Then keys 0 and 2', 'new m=ln(2); old multiplier exp(0-ln(2))=0.5'),
        ('Updated denominator', 'l = 0.5*1 + 1 + 1 = 2.5'),
        ('Updated numerator', 'z = 0.5*[0,2] + [1,0] + [3,4] = [4,5]')], [230, 474])
    d.note('Readiness and reuse are separate',
           'A score tile must be ready before its softmax consumer. A V tile and its probability operands must remain live until PV readers finish. Retaining row state does not permit overwriting active operand storage.', 'amber')
    d.note('Mask contract changes the mathematics',
           'With explicitly top-left causal j <= i, row 0 retains only key 0 and returns [1,0]; row 1 retains keys 0 and 1 and returns [1/3,4/3]. This rectangular-mask convention is stated, not inferred.', 'blue')
    assert math.isclose((2*1+1*0+2*3)/5, 1.6)
    assert math.isclose((1*1+2*0+2*3)/5, 1.4)
    d.write()


def ownership():
    d = canvas(31, 'fmha-ownership', 'Allocate Q, K, V and outputs; stream the intermediates',
               'Pinned Hopper FMHA validation shape: B=1, H=2, Q=128, K=128, D=64. Counts below are logical tensor payload, not register allocation or measured traffic.')
    d.section('1  Audit the input and output byte counts')
    d.table(['Tensor', 'Shape / type', 'Elements', 'Bytes'], [
        ('Q', '128 x 64 x (1,2), FP16', '16,384', '32,768'),
        ('K', '128 x 64 x (1,2), FP16', '16,384', '32,768'),
        ('V', '128 x 64 x (1,2), FP16', '16,384', '32,768'),
        ('O', '128 x 64 x (1,2), FP16', '16,384', '32,768'),
        ('LSE', '128 x (1,2), FP32', '256', '1,024')], [95, 315, 145, 149])
    d.note('Every stride has a unit',
           'D is contiguous in these logical views. A half element is 2 bytes; an FP32 LSE element is 4 bytes. The head/batch stride contract still comes from the actual layout, not from a flattened rank number.', 'blue')
    d.section('2  A tile dependency is not a physical time interval')
    d.flow([
        ('TMA: global operands to shared tiles', 'Observe the matching data-ready generation before reading Q/K/V storage. Which operand tile is reused depends on the selected schedule.', 'blue'),
        ('QK: accumulate the score tile', 'WGMMA produces FP32 QK accumulators. Apply the selected logical-coordinate mask. The pinned row maximum stays in unscaled QK units; exponential evaluation applies the scale.', 'violet'),
        ('Online row update', 'Track the raw QK maximum, denominator l and output numerator. Rescale prior contributions with exp2((old_max-new_max)*scale_softmax_log2) when the maximum changes.', 'teal'),
        ('PV: accumulate weighted values', 'Consume probability operands and V tiles with their required layout/precision. Keep each shared stage live until its asynchronous readers finish.', 'violet'),
        ('Epilogue: normalized O and LSE', 'Normalize O and store natural-log LSE = raw_max*scale_softmax + ln(l). Raw maximum units differ from Chapter 30’s scaled-score m. No full global Q-by-K score buffer is required.', 'teal')])
    d.section('3  Compare schedules without inventing a timeline')
    d.table(['Variant', 'CTA tile M x N x D', 'Logical query rows'], [
        ('TMA', '64 x 128 x 64', '64 per tile'),
        ('Cooperative', '128 x 64 x 64', '128 per tile'),
        ('Ping-pong', '128 x 64 x 64', '128 per tile')], [220, 270, 214])
    d.note('A counterfactual global intermediate budget',
           'A full FP32 score matrix for both heads would need 2*128*128 = 32,768 elements, or 131,072 bytes. This is not an allocated buffer here: the fused kernel avoids it, and the pinned reference uses per-query-row shared score storage rather than a full global score matrix.', 'amber')
    assert 128*64*2*2 == 32768 and 128*2*4 == 1024
    d.write()


if __name__ == '__main__':
    softmax()
    ownership()
    print('Generated attention worked figures for Chapters 30 and 31.')
