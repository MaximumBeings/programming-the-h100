#!/usr/bin/env python3
"""Original chapter maps and worked inference-system diagrams (no timing claims)."""
from pathlib import Path
from html import escape
import re
from illustrations.drawing import Drawing, FONT_STYLE, LICENSE, wrap
ROOT=Path(__file__).resolve().parents[1]
# Chapter, label, subtitle, three actual mechanisms, worked hand trace.
DATA=[
(32,'Measure before tuning','A request becomes a timeline, then a defensible result.',('Arrival','First token','Last token'),('0.00 s: request arrives; 0.35 s: first token; 1.05 s: last token.','Eight output tokens give TTFT=0.35 s and TPOT=(1.05-0.35)/7=0.10 s.','Count seven inter-token gaps, not eight. Queue delay belongs in TTFT.')),
(33,'Route. Compute. Combine.','Expert ownership determines where token copies travel.',('Top-k router','Expert owners','Weighted sum'),('Token x selects expert 1 with weight 0.7 and expert 3 with weight 0.3.','Dispatch x to both owners; receive expert outputs y1 and y3.','Return 0.7*y1+0.3*y3 to the source token position; preserve its identity.')),
(34,'Memory has a hierarchy','Moving KV is useful only when its cost is accounted for.',('GPU blocks','Host RAM','NVMe tiers'),('A 70B-shaped GQA cache uses 2*80*8*128*2 = 327,680 bytes per token.','An 8,192-token sequence therefore needs 2.5 GiB of BF16 KV.','For the whole model on one assumed 55 GB/s path, restoring KV takes at least 48.8 ms, before overhead.')),
(35,'Split work, count transfers','Prefill and decode can live apart; KV must cross the boundary.',('Prefill worker','KV transfer','Decode worker'),('The prompt has 8,192 tokens and 327,680 KV bytes per token.','Transfer volume is 2,684,354,560 bytes; at assumed 50 GB/s the floor is 53.7 ms.','Compare saved interference with exposed transfer, queues and cost. A layerwise pipeline still leaves a final-layer transfer tail.')),
(36,'Benchmark the whole path','A load generator must count failures as carefully as successes.',('Open-loop load','Streaming events','SLO + goodput'),('Send request at intended time 0.00 s, actually send at 0.10 s, first token at 0.40 s.','Intended-time TTFT is 0.40 s; sent-time TTFT is only 0.30 s.','The 0.10 s difference is client scheduling delay. Publish both to expose overload.')),
(37,'Propose, verify, accept','Speculation trades extra work for fewer target-model steps.',('Draft tokens','Target verify','Accept / repair'),('Assume K=3 draft tokens and independent conditional acceptance alpha=0.8.','Expected emitted tokens = 1+0.8+0.64+0.512 = 2.952 per verification cycle.','Speedup is 2.952*baseline_step/cycle_time; it is not automatically 2.952x.')),
(38,'A similar prompt can differ','A semantic hit is a quality decision, not just a vector lookup.',('Scope + TTL','Similarity gate','Validate answer'),('Two questions differ only in a date or a negation; cosine similarity may remain high.','Partition by tenant, model revision and generation settings before searching.','Calibrate false hits on held-out labelled data; 0.98 similarity is not 98% correctness.')),
(39,'Route with a quality budget','A cheaper model helps only while the quality constraint holds.',('Small-model probe','Calibrated policy','Small / large'),('For probabilities [0.5,0.5], entropy is ln(2); for [1,0], its limiting value is zero.','Entropy describes uncertainty under this model, not objective query difficulty.','Evaluate router and both models on the same held-out examples, including probe cost.')),
(40,'Scale the queue responsibly','Demand, readiness and draining happen on different clocks.',('Pending work','Ready replicas','Drain + release'),('Assume 900 queued tokens and a target of 300 queued tokens per replica.','The raw desired count is ceil(900/300)=3 before bounds and stabilization rules.','A requested replica is not ready capacity. Include startup time and prevent retry storms.')),
(41,'Fail over with limits','A fallback needs capacity, compatible behaviour and a stop rule.',('Circuit breaker','Retry budget','Fallback tier'),('A failed request gets at most two retries with jitter and an overall deadline.','An open breaker fails fast; a bounded half-open probe tests recovery.','Fallback may change quality and context limits; do not replay side effects blindly.')),
(42,'One gateway, shared metrics','Cache, routing, scaling and resilience interact.',('Cache + router','Capacity pools','Fallback policy'),('A cache hit avoids model execution but can return the wrong answer.','A cache miss enters routing; the selected pool can still queue or fail.','Count final quality, failures, all billed replicas and every request in one record.')),
(43,'Logical tokens, physical blocks','Block tables allow sparse allocation and safe sharing.',('Token position','Block table','Physical KV'),('With block size B=4 and table [7,2], logical token 5 uses table[5//4]=2.','Its slot is 5%4=1: physical block 2, slot 1, for both K and V.','If the last block is shared, copy it before appending into its unused slots.')),
(44,'Refill at every iteration','Admission and preemption must obey the same memory budget.',('Admit + budget','Prefill / decode','Release / repeat'),('With a 12-token iteration budget, reserve 3 tokens for three active decodes.','At most 9 prompt tokens fit in this simplified shared token budget.','Real scheduler policies can differ; never silently oversubscribe blocks during admission.')),
(45,'Reuse the exact prefix','Token-prefix reuse differs from semantic response caching.',('Prefix key','Pinned KV blocks','Evictable leaves'),('Prompt A=[1,2,3,4,5] and prompt B=[1,2,3,9] share exactly [1,2,3].','With whole blocks of size 2, only the first 2 tokens are reusable.','Include model, adapter, tenant and other KV-affecting context in a production cache key.')),
(46,'Build the engine core','The scheduler drives a real tiny transformer through paged KV.',('Schedule chunks','Paged forward','Reference check'),('A prompt of length 9 and B=4 needs ceil(9/4)=3 blocks.','Only complete prefix blocks may be shared; the unfinished final block stays private.','Compare every generated token with dense no-cache decoding; separately measure speed.')),
(47,'Tune a working server','A pinned version and fixed workload make the experiment repeatable.',('Baseline server','One setting','Compare evidence'),('Fix model, revision, prompt, output length and GPU before changing the batch cap.','Record first-token time, completion-token count, elapsed time and failures.','Repeat and vary offered load; one short request demonstrates execution, not peak throughput.')),
]

def banner(n,title,lede,nodes):
    parts=[]
    for i,line in enumerate(wrap(title,275,31)):
        parts.append(f'<text x="38" y="{115+i*39}" font-size="31" font-weight="700" fill="#e8f5f3">{escape(line)}</text>')
    for i,line in enumerate(wrap(lede,270,18)):
        parts.append(f'<text x="38" y="{298+i*26}" font-size="18" fill="#a8c6c8">{escape(line)}</text>')
    for i,(node,color) in enumerate(zip(nodes,('#2dd4bf','#fbbf24','#a3e635'))):
        y=83+i*108
        parts.append(f'<rect x="346" y="{y}" width="376" height="77" rx="14" fill="#12343a" stroke="{color}" stroke-width="1.8"/>')
        # A small ownership tile: a conceptual resource, not a GPU die layout.
        for row in range(3):
            for col in range(3):
                parts.append(f'<rect x="{363+col*11}" y="{y+23+row*11}" width="8" height="8" rx="2" fill="{color}"/>')
        parts.append(f'<text x="418" y="{y+45}" font-size="21" fill="#e8f5f3">{escape(node)}</text>')
        if i<2:parts.append(f'<path d="M534 {y+78} v26 l-5 -7 m5 7 l5 -7" stroke="{color}" stroke-width="2.5" fill="none"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="450" viewBox="0 0 760 450" role="img" aria-labelledby="title desc"><title id="title">{escape(str(n))}: {escape(title)}</title><desc id="desc">{escape(lede)} Components: {escape(', '.join(nodes))}. Conceptual illustration.</desc>{LICENSE}{FONT_STYLE}<defs><pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#23616a" opacity="0.36"/></pattern></defs><rect width="760" height="450" rx="16" fill="#07191c"/><rect width="760" height="450" rx="16" fill="url(#dots)"/><g font-family="BookDiagram, Inter, Arial, sans-serif"><path d="M22 79 v294" stroke="#a3e635" stroke-width="4" stroke-linecap="round"/><text x="38" y="45" font-size="15" fill="#a3e635" font-weight="700">CHAPTER {escape(str(n))} / INFERENCE ENGINEERING</text>{''.join(parts)}<text x="722" y="423" font-size="13" fill="#a8c6c8" text-anchor="end">CONCEPTUAL ILLUSTRATION</text></g></svg>'''


def main():
    for n,title,lede,nodes,trace in DATA:
        path=f'worked/ch{n}-inference.svg'
        d=Drawing(f'ch{n}/01-inference.svg',title,lede,'Original inference-system teaching schematic. Arithmetic uses the assumptions stated in the labels.')
        d.path=path
        d.columns([(s,b,c) for s,b,c in zip(nodes,trace,('teal','blue','amber'))],n=1)
        d.note('What this illustration establishes','Follow the ownership, state and accounting. This is a worked conceptual trace, not a measured GPU timeline.')
        d.write()
        (ROOT/f'docs/images/chapter-openers/{n}-inference.svg').write_text(banner(n,title,lede,nodes))
        if n<47:
            p=next((ROOT/'docs').glob(f'{n}-*.md'));s=p.read_text();first=s.find('\n')
            if f'{n}-inference.svg' not in s:s=s[:first]+f'\n\n![Chapter {n}: {title}](images/chapter-openers/{n}-inference.svg)\n'+s[first:]
            marker='\n## '
            pos=s.find(marker)
            if f'ch{n}-inference.svg' not in s:s=s[:pos]+f'\n## Worked system map\n\n![{title}: step-by-step hand trace](images/worked/ch{n}-inference.svg)\n\n'+s[pos:]
            p.write_text(s)
def appendix():
    d=Drawing('chA/01-fp4.svg','Two FP4 formats, two scaling paths',
              'Both store E2M1 elements. Scaling metadata changes reconstruction and storage cost.',
              'Worked CPU-emulation arithmetic; no native H100 FP4 instruction or B200 timing claim.')
    d.path='worked/appendix-fp4.svg'
    d.columns([('MXFP4: 32 elements per scale','32*4+8 = 136 bits per full block; 136/32 = 4.25 bits per element. A shared E8M0 scale is a power of two.','blue'),
               ('NVFP4: 16 elements per scale','16*4+8 = 72 bits per full block; 72/16 = 4.5 bits per element, plus an FP32 tensor-scale overhead.','teal')])
    d.flow([('Normalize one value','With block scale 0.46875 and tensor scale 1, x=-2.3 becomes approximately -4.907.','blue'),
            ('Round its E2M1 element','The nearest magnitudes are 4 and 6. The value is closer to 4, so the signed element is -4.','teal'),
            ('Reconstruct and compare','-4*0.46875*1 = -1.875. Reconstruction error is +0.425. This is quantization error, not a GEMM layout error.','amber')])
    d.note('Separate two references','Compare a kernel with the same quantized codes/scales to find layout or arithmetic bugs; compare with original values separately to assess quantization loss.')
    d.write()
    (ROOT/'docs/images/chapter-openers/appendix-fp4.svg').write_text(
        banner('FP4','Elements. Scales. Reconstruction.',
               'Quantization changes values. A reference tells us which error comes from the format.',
               ('E2M1 elements','Shared block scales','Quantized reference')))
    d=Drawing('ch00/01-inference-roadmap.svg','Build an engine by replacing verified pieces',
              'Use a simple reference as the correctness anchor before optimizing or distributing it.',
              'Original cross-book learning roadmap; not a claim of complete production implementation.')
    d.path='worked/inference-roadmap.svg'
    d.flow([('H100: establish ownership','Trace request timing, physical KV blocks, iteration scheduling and exact prefix reuse.','teal'),
            ('H100: execute the baseline','Run CPU references, CUDA numerical probes and a real small-model serving client. Preserve their separate scopes.','blue'),
            ('Triton: replace one kernel','Compare against the reference, test edge cases and profile the memory and compute path.','teal'),
            ('Multi-GPU: distribute state','Add collectives, topology-aware ownership and failure handling; rerun distributed correctness tests.','amber')])
    d.write()

if __name__=='__main__':
    main()
    appendix()
    for page in (ROOT/'docs').glob('*.md'):
        source=page.read_text()
        source=re.sub(r'(?m)^!\[([^\]]*)\]\((images/[^\n)]*(?:-inference|inference-roadmap|appendix-fp4)\.svg)\)$',
                      r'[![\1](\2)](\2)', source)
        page.write_text(source)
