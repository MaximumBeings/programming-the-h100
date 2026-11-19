"""Original Blackwell appendix diagrams; logical ownership and arithmetic only."""
from illustrations.drawing import Drawing

def figure(number,title,lede):
    d=Drawing(f'chB200/{number:02}-blackwell.svg',title,lede,
              'Original teaching schematic. CUTLASS v4.3.6 SM100 implementations; no physical timing or allocation measurement.')
    d.parts=[part.replace('CHAPTER B200','B200 APPENDIX') for part in d.parts]
    d.path=f'worked/blackwell-b200-{number:02}.svg'
    return d

def main():
    d=figure(1,'From Hopper WGMMA to Blackwell UMMA','Keep the mathematics; rebuild instruction, accumulator-storage and completion contracts for SM100.')
    d.columns([('Hopper / SM90','WGMMA uses warpgroup matrix operations and register accumulator fragments. The book uses architecture-specific sm_90a kernels.','violet'),('B200 / SM100','tcgen05 UMMA uses Tensor Memory for accumulators. These native examples build with sm_100a and Sm100 collectives.','teal')])
    d.section('One native GEMM, five ownership stages')
    d.flow([('Global tensors','A and B are FP16; C and D are FP32. The CPU oracle uses the same packed matrix coordinates.','blue'),('TMA produces shared operand tiles','Do not read A/B shared storage before its matching data-ready completion.','teal'),('UMMA produces Tensor Memory accumulators','The selected SM100 collective owns TMEM allocation and asynchronous MMA completion.','violet'),('Epilogue consumes completed accumulators','Compute alpha*AB + beta*C, optionally apply ReLU, then store the output.','amber'),('Validate every output','NaN sentinels expose missed stores. Three shapes cover ordinary tiles, batching and tails.','blue')])
    d.note('Two lifetimes, not one','A shared operand stage may be reused only after its readers finish. TMEM accumulator storage must remain live until its epilogue consumer finishes. A Hopper wait-group sequence is not a Blackwell completion protocol.','red')
    d.write()
    d=figure(2,'Calculate the GEMM before choosing the kernel','A derived 2x2 example; recorded native GPU tests use the larger shapes listed in the appendix.')
    d.table(['Matrix','Rows'],[('A','[1,2]; [3,4]'),('B','[5,-6]; [7,-8]'),('C','[2,-2]; [4,-4]'),('Scalars','alpha=1/2; beta=1/4')],[150,554])
    d.section('Trace every intermediate')
    d.table(['Stage','Row 0','Row 1'],[('A*B','[19,-22]','[43,-50]'),('alpha*A*B','[9.5,-11]','[21.5,-25]'),('beta*C','[0.5,-0.5]','[1,-1]'),('Linear output','[10,-11.5]','[22.5,-26]'),('ReLU output','[10,0]','[22.5,0]')],[210,247,247])
    d.note('Check one coordinate','D[0,1] = (1*(-6)+2*(-8))/2 + (-2)/4 = -11.5. ReLU changes this value to zero; it does not change the reduction itself.','teal')
    d.section('A real 128x128x64 logical tile')
    d.table(['Payload','Calculation','Bytes'],[('FP16 A tile','128*64*2','16,384'),('FP16 B tile','64*128*2','16,384'),('FP32 accumulator values','128*128*4','65,536')],[250,250,204])
    d.note('Logical payload is not an allocation report','Descriptors, stage count, alignment, TMEM layout and epilogue storage change physical resource use. Query the selected kernel rather than treating these sums as shared-memory reservations.','amber')
    d.section('The tested tail case')
    d.table(['Dimension','Last tile base','Valid extent'],[('M=136, tile M=128','128','8 rows'),('N=264, tile N=128','256','8 columns'),('K=72, tile K=64','64','8 reduction terms')],[300,202,202])
    d.note('Check the entire result','The oracle visits all 35,904 outputs of the tail shape, not only tile boundaries or a checksum.','blue')
    d.write()
    d=figure(3,'Attention changes storage, not its mathematical answer','Recorded native forward cases: B=1, Hq=2, Q=K=256; FP16 inputs/output, FP32 LSE, D=64 or 128.')
    d.flow([('QK and masking','Compute scaled dot products; full attention keeps all keys, top-left causal attention retains j <= i.','blue'),('Online softmax state','Update the row maximum, shifted denominator and weighted numerator; rescale old state when its maximum changes.','teal'),('PV and normalization','Multiply probability weights by V and normalize the output. Native SM100 collectives manage TMEM and pipeline completion.','violet'),('Independent implementation check','The upstream GPU reference checks every O and LSE value under its FP16 tolerances. This is not a CPU/FP64 oracle.','amber')])
    d.section('D=64 logical tensor payload')
    d.table(['Tensor','Calculation','Bytes'],[('Each of Q, K and V','1*2*256*64*2','65,536'),('Output O','1*2*256*64*2','65,536'),('Natural-log LSE','1*2*256*4','2,048'),('Counterfactual FP32 scores','1*2*256*256*4','524,288')],[250,250,204])
    d.note('The score matrix is counterfactual','The fused forward algorithm streams score/probability tiles instead of allocating that complete global score tensor. These are payload counts, not measured HBM traffic.','teal')
    d.section('GQA: two query heads share one KV head')
    d.table(['Quantity','Hkv=2','Hkv=1'],[('Q bytes','65,536','65,536'),('Combined K+V bytes','131,072','65,536'),('O bytes','65,536','65,536')],[300,202,202])
    d.note('A storage saving is not a speedup result','GQA halves K/V storage in this example. It does not halve query/output size or establish twice the performance.','red')
    d.write()

if __name__=='__main__':main()
