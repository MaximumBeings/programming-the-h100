#!/usr/bin/env python3
"""CPU arithmetic/ownership oracles. No CUDA, CuTe kernel or profiler is executed."""
from collections import Counter
import math


def check(condition, message):
    if not condition: raise AssertionError(message)


def layouts():
    row = [[m*4+n for n in range(4)] for m in range(3)]
    column = [[m+n*3 for n in range(4)] for m in range(3)]
    check(row[2][3] == column[2][3] == 11, 'last element')
    check(row[1][2] == 6 and column[1][2] == 7, 'major convention')
    offsets = [a+3*b+40*c for c in range(2) for b in range(4) for a in range(3)]
    check(len(set(offsets)) == 24 and max(offsets)+1 == 52, 'hierarchy span')
    check(2+3*3+40 == 51, 'hierarchical coordinate')
    padded = [m*8+n for m in range(3) for n in range(4)]
    check(len(padded) == 12 and max(padded)+1 == 20, 'padded span')
    print('LAYOUTS PASS: row(1,2)=6 column(1,2)=7; nested size=24 span=52 holes=28')


def cta_offsets():
    M,N,K,BM,BN,BK,bm,bn = 4096,2048,1024,128,128,64,5,3
    m0,n0 = bm*BM,bn*BN
    a,b,c = m0*K,n0*K,m0*N+n0  # Row A, column B, row C.
    check((m0,n0,a,b,c) == (640,384,655360,393216,1311104),'CTA base')
    check(a+BK == 655424 and b+BK == 393280,'K origin')
    M,BM=3100,64
    blocks=(M+BM-1)//BM;last=(blocks-1)*BM
    check((blocks,last,M-last)==(49,3072,28),'ceil boundary')
    print('CTA PASS: origin=(640,384) A=655360 B=393216 C=1311104; boundary grid=49 valid_rows=28')


def tiled_ownership():
    M,N,TM,TN = 37,70,4,32
    seen=Counter(); invalid=0
    for bm in range((M+TM-1)//TM):
        for bn in range((N+TN-1)//TN):
            for tid in range(128):
                rm,rn=tid//32,tid%32
                m,n=bm*TM+rm,bn*TN+rn
                if m<M and n<N:seen[m,n]+=1
                else:invalid+=1
    check(len(seen)==M*N and set(seen.values())=={1},'copy coverage')
    check(invalid==10*3*128-M*N,'predicated tail')
    print(f'COPY OWNERSHIP PASS: {len(seen)} valid coordinates owned once; {invalid} invalid lanes predicated')


def fragments():
    seen=Counter()
    for lane in range(32):
        group,thread=lane//4,lane%4
        for reg in range(4):
            row=group+8*(reg//2);col=2*thread+reg%2
            seen[row,col]+=1
    check(len(seen)==16*8 and set(seen.values())=={1},'m16n8 FP32 ownership')
    lane=10;g,t=lane//4,lane%4
    coords=[(g+8*(r//2),2*t+r%2) for r in range(4)]
    check(coords==[(2,4),(2,5),(10,4),(10,5)],'lane 10')
    print('FRAGMENTS PASS: 32 lanes x 4 FP32 values cover 16x8 exactly once; lane10='+str(coords))


def swizzles():
    normal=[i*32+j for i in range(32) for j in range(32)]
    row_xor=[(i^(j>>2))*32+j for i in range(32) for j in range(32)]
    col_xor=[i*32+(j^i) for i in range(32) for j in range(32)]
    check(len(set(row_xor))==len(set(col_xor))==1024,'swizzle bijection')
    fixed_col=7
    banks=lambda f:{f(i,fixed_col)%32 for i in range(32)}
    check(len(banks(lambda i,j:i*32+j))==1,'unswizzled column')
    check(len(banks(lambda i,j:(i^(j>>2))*32+j))==1,'wrong XOR benefit')
    check(len(banks(lambda i,j:i*32+(j^i)))==32,'word XOR benefit')
    check(normal!=row_xor and normal!=col_xor,'nontrivial permutation')
    print('SWIZZLE PASS: 32-bit fixed-column distinct banks: plain=1 row-XOR=1 column-XOR=32')


def pipeline():
    for stages in (1,2,3):
        for tiles in (0,1,2,3,7):
            state=['empty']*stages;generation=[0]*stages
            for tile in range(tiles):
                index=tile%stages;expected=tile//stages
                check(state[index]=='empty' and generation[index]==expected,'acquire generation')
                state[index]='writing';state[index]='full';state[index]='reading'
                # This model releases only after every modeled reader completes.
                state[index]='empty';generation[index]+=1
            check(set(state)=={'empty'},'pipeline drain')
    print('PIPELINE MODEL PASS: 15 depth/tile combinations; acquire, ready, read, release and generation checks')


def numerics():
    A=[[1,-2,3],[4,0,-1]];B=[[2,1],[3,-2],[1,4]];C=[[2,-4],[6,8]]
    product=[[sum(A[m][k]*B[k][n] for k in range(3)) for n in range(2)] for m in range(2)]
    D=[[0.5*product[m][n]+0.25*C[m][n] for n in range(2)] for m in range(2)]
    check(product==[[-1,17],[7,0]] and D==[[0,7.5],[5,2]],'GEMM reference')
    ops=2*2048*2048*4096;rate=ops/(85e-6)/1e12
    check(math.isclose(rate,404.23221609411764),'operation count')
    print(f'NUMERICS PASS: AB={product} D={D}; 34.359738368 GFLOP / 85 us = {rate:.3f} TFLOP/s (arithmetic example)')

if __name__=='__main__':
    for fn in (layouts,cta_offsets,tiled_ownership,fragments,swizzles,pipeline,numerics):fn()
    print('WORKBOOK PASS: 7 CPU-only reference groups; no GPU execution or performance measurement')
