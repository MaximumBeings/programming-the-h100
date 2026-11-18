// ex05_ptx_datatypes.cu — PTX data types: FP16/BF16/FP64/packed ops

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(call) do { \
    cudaError_t _e=(call); \
    if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n", \
    __FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} } while(0)

// ── Test 1: FP32 → FP16 conversion ───────────────────────────────
// cvt.rn.f16.f32: convert with round-to-nearest.
// "h" = 16-bit register constraint for FP16.
__global__ void test_fp32_to_fp16(uint16_t* out, float val) {
    uint16_t result;
    asm("cvt.rn.f16.f32 %0, %1;"   // round-to-nearest FP32→FP16
        : "=h"(result)              // 16-bit output register
        : "f"(val));                // 32-bit float input
    out[0] = result;
}

// ── Test 2: FP16 pack / unpack roundtrip ─────────────────────────
// Packs two FP16 values into one 32-bit register and unpacks them.
// This is the .f16x2 format that tensor cores require.
__global__ void test_fp16_pack(uint32_t* out, float a, float b) {
    uint16_t ha, hb;
    asm("cvt.rn.f16.f32 %0,%1;" : "=h"(ha) : "f"(a));
    asm("cvt.rn.f16.f32 %0,%1;" : "=h"(hb) : "f"(b));

    // Pack: ha → bits[15:0], hb → bits[31:16]
    uint32_t packed;
    asm("mov.b32 %0, {%1,%2};" : "=r"(packed) : "h"(ha), "h"(hb));

    // Unpack back
    uint16_t ua, ub;
    asm("mov.b32 {%0,%1}, %2;" : "=h"(ua), "=h"(ub) : "r"(packed));

    out[0] = packed;
    out[1] = ua;   // should equal ha (a's FP16 bits)
    out[2] = ub;   // should equal hb (b's FP16 bits)
}

// ── Test 3: Packed BF16 add ───────────────────────────────────────
// BF16 is stored as raw bits (.b16) in PTX; arithmetic uses .bf16x2.
// add.bf16x2 adds two BF16 pairs, same pattern as add.f16x2.
__global__ void test_bf16_add(uint32_t* out, float a, float b) {
    // __nv_bfloat16 is the C++ type; we work with its raw uint16 bits
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);

    uint16_t ra = *reinterpret_cast<uint16_t*>(&ba);
    uint16_t rb = *reinterpret_cast<uint16_t*>(&bb);

    // Pack two BF16 values
    uint32_t pa, pb;
    asm("mov.b32 %0, {%1,%2};" : "=r"(pa) : "h"(ra), "h"(ra));  // both slots = a
    asm("mov.b32 %0, {%1,%2};" : "=r"(pb) : "h"(rb), "h"(rb));  // both slots = b

    // Packed BF16 add: adds both pairs simultaneously
    uint32_t pr;
    asm("add.bf16x2 %0, %1, %2;" : "=r"(pr) : "r"(pa), "r"(pb));

    // Convert result back to float for verification
    uint16_t r_lo, r_hi;
    asm("mov.b32 {%0,%1}, %2;" : "=h"(r_lo), "=h"(r_hi) : "r"(pr));
    __nv_bfloat16 res_lo, res_hi;
    *reinterpret_cast<uint16_t*>(&res_lo) = r_lo;
    *reinterpret_cast<uint16_t*>(&res_hi) = r_hi;

    out[0] = __float_as_uint(__bfloat162float(res_lo));   // a+b in lo slot
    out[1] = __float_as_uint(__bfloat162float(res_hi));   // a+b in hi slot
}

// ── Test 4: FP64 addition ─────────────────────────────────────────
// FP64 uses register PAIRS in PTX (two 32-bit slots = 64 bits).
// "d" constraint maps to .f64 (register pair).
__global__ void test_fp64(double* out, double a, double b) {
    double result;
    asm("add.f64 %0, %1, %2;"
        : "=d"(result)        // "d" = 64-bit float (register pair)
        : "d"(a), "d"(b));
    out[0] = result;
}

// ── Test 5: Integer MAD with overflow detection ───────────────────
// mad.lo.u32 computes (a*b+c) taking low 32 bits.
// mad.hi.u32 computes (a*b+c) taking high 32 bits.
// Together they give the full 64-bit result.
__global__ void test_mad_full(uint32_t* out, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t lo, hi;
    asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(lo) : "r"(a),"r"(b),"r"(c));
    asm("mad.hi.u32 %0, %1, %2, 0;"  : "=r"(hi) : "r"(a),"r"(b));
    // Together: full 64-bit = (uint64_t)hi<<32 | lo
    out[0] = lo;
    out[1] = hi;
}

// Copy the representation without aliasing or device-only intrinsics.
static float host_uint_as_float(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  ex05: PTX Data Types & Packed Operations             ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    int tests=0, passed=0;

    // Test 1: FP32→FP16
    printf("Test 1: cvt FP32→FP16 (1.5f → 0x3E00)\n");
    {
        uint16_t *d,h;
        CHECK_CUDA(cudaMalloc(&d,2));
        test_fp32_to_fp16<<<1,1>>>(d,1.5f);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&h,d,2,cudaMemcpyDeviceToHost));
        tests++;
        if(h==0x3E00){passed++; printf("  ✓ PASS: 0x%04X\n",h);}
        else printf("  ✗ FAIL: 0x%04X (expected 0x3E00)\n",h);
        cudaFree(d);
    }

    // Test 2: FP16 pack/unpack
    printf("\nTest 2: FP16 pack/unpack (1.0f=0x3C00, 2.0f=0x4000)\n");
    {
        uint32_t *d,h[3];
        CHECK_CUDA(cudaMalloc(&d,12));
        test_fp16_pack<<<1,1>>>(d,1.0f,2.0f);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h,d,12,cudaMemcpyDeviceToHost));
        // lo=0x3C00 (1.0f), hi=0x4000 (2.0f), packed=0x40003C00
        tests++;
        if((uint16_t)h[1]==0x3C00 && (uint16_t)h[2]==0x4000){
            passed++;
            printf("  ✓ PASS: packed=0x%08X lo=0x%04X hi=0x%04X\n",
                   h[0],(uint16_t)h[1],(uint16_t)h[2]);
        } else printf("  ✗ FAIL: lo=0x%04X hi=0x%04X\n",(uint16_t)h[1],(uint16_t)h[2]);
        cudaFree(d);
    }

    // Test 3: BF16 packed add
    printf("\nTest 3: add.bf16x2 (2.0+3.0=5.0 in both slots)\n");
    {
        uint32_t *d,h[2];
        CHECK_CUDA(cudaMalloc(&d,8));
        test_bf16_add<<<1,1>>>(d,2.0f,3.0f);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h,d,8,cudaMemcpyDeviceToHost));
        float r0=host_uint_as_float(h[0]), r1=host_uint_as_float(h[1]);
        tests++;
        if(fabsf(r0-5.0f)<0.05f && fabsf(r1-5.0f)<0.05f){
            passed++; printf("  ✓ PASS: lo=%.2f hi=%.2f\n",r0,r1);
        } else printf("  ✗ FAIL: lo=%.4f hi=%.4f\n",r0,r1);
        cudaFree(d);
    }

    // Test 4: FP64
    printf("\nTest 4: add.f64 (1.23456789012345 + 9.87654321098765)\n");
    {
        double *d,h;
        CHECK_CUDA(cudaMalloc(&d,8));
        test_fp64<<<1,1>>>(d,1.23456789012345,9.87654321098765);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&h,d,8,cudaMemcpyDeviceToHost));
        double expected=1.23456789012345+9.87654321098765;
        tests++;
        if(fabs(h-expected)<1e-12){passed++; printf("  ✓ PASS: %.15f\n",h);}
        else printf("  ✗ FAIL: %.15f expected %.15f\n",h,expected);
        cudaFree(d);
    }

    // Test 5: MAD hi+lo (full 64-bit product)
    // 0xFFFFFFFF * 0xFFFFFFFF + 0 = 0xFFFFFFFE00000001
    // lo = 0x00000001, hi = 0xFFFFFFFE
    printf("\nTest 5: mad hi+lo (0xFFFFFFFF * 0xFFFFFFFF)\n");
    {
        uint32_t *d,h[2];
        CHECK_CUDA(cudaMalloc(&d,8));
        test_mad_full<<<1,1>>>(d,0xFFFFFFFF,0xFFFFFFFF,0);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h,d,8,cudaMemcpyDeviceToHost));
        tests++;
        if(h[0]==0x00000001 && h[1]==0xFFFFFFFE){
            passed++;
            printf("  ✓ PASS: lo=0x%08X hi=0x%08X\n",h[0],h[1]);
        } else printf("  ✗ FAIL: lo=0x%08X hi=0x%08X\n",h[0],h[1]);
        cudaFree(d);
    }

    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  %d/%d tests passed %-34s║\n",passed,tests,
           passed==tests?"  ✓ ALL PASS":"  ✗ FAILURES");
    printf("╚══════════════════════════════════════════════════════╝\n");
    return (passed==tests)?0:1;
}
