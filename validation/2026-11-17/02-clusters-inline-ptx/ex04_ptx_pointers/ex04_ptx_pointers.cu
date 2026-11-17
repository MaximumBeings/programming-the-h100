// ex04_ptx_pointers.cu — State spaces, cvta, packed FP16, bit reinterpret

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr,"CUDA error %s:%d %s\n",__FILE__,__LINE__, \
                cudaGetErrorString(_e)); exit(1); } } while(0)

// ── Test 1: cvta generic→shared + ld.shared ───────────────────────
// Writes 42.0f and 99.0f via C++, reads back via explicit PTX ld.shared.
// Proves that __cvta_generic_to_shared gives a usable 32-bit offset.
__global__ void test_cvta_shared(uint32_t* results) {
    __shared__ float smem[32];
    if (threadIdx.x != 0) return;

    smem[0] = 42.0f;
    smem[1] = 99.0f;

    // Strip tag bits from the C++ pointer → 32-bit SMEM offset
    uint32_t addr = __cvta_generic_to_shared(&smem[0]);

    float v0, v1;
    asm volatile("ld.shared.f32 %0, [%1];"   : "=f"(v0) : "r"(addr));
    asm volatile("ld.shared.f32 %0, [%1+4];" : "=f"(v1) : "r"(addr));
    // [addr+4] = next float (4 bytes further in SMEM)

    results[0] = __float_as_uint(v0);  // 42.0f bits
    results[1] = __float_as_uint(v1);  // 99.0f bits
    results[2] = addr;                 // raw SMEM byte offset
}

// ── Test 2: st.shared + ld.shared with thread-indexed addresses ───
// Thread i stores (i*10+7) to smem[i].
// Thread i then reads smem[(i+1)%N] — the circular neighbor.
// Thread 0 reports its own value (7.0) and neighbor's value (17.0).
__global__ void test_explicit_shared(float* results) {
    __shared__ float smem[64];

    uint32_t base = __cvta_generic_to_shared(smem);
    uint32_t my_addr = base + threadIdx.x * sizeof(float);

    float my_val = threadIdx.x * 10.0f + 7.0f;
    asm volatile("st.shared.f32 [%0], %1;"
                 : : "r"(my_addr), "f"(my_val) : "memory");
    __syncthreads();

    uint32_t nb_addr = base + ((threadIdx.x + 1) % blockDim.x) * sizeof(float);
    float nb_val;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(nb_val) : "r"(nb_addr));

    if (threadIdx.x == 0) { results[0] = my_val; results[1] = nb_val; }
}

// ── Test 3: ld.global / st.global with 64-bit addresses ──────────
// Multiplies each element by 2.0 using explicit PTX global loads/stores.
// KEY: global pointers are 64-bit → use "l" constraint, NOT "r"!
__global__ void test_global_ptx(float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint64_t ia = reinterpret_cast<uint64_t>(in)  + (uint64_t)i * 4;
    uint64_t oa = reinterpret_cast<uint64_t>(out) + (uint64_t)i * 4;

    float v;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(ia));
    float r;
    asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(v), "f"(2.0f));
    asm volatile("st.global.f32 [%0], %1;" : : "l"(oa), "f"(r) : "memory");
}

// ── Test 4: atom.shared.add — all threads increment shared counter ─
// 64 threads, each atomically adds 1 → final counter = 64.
// atom.shared returns the OLD value; we check the final counter.
__global__ void test_shared_atomic(uint32_t* result) {
    __shared__ uint32_t counter;
    if (threadIdx.x == 0) counter = 0;
    __syncthreads();

    uint32_t saddr = __cvta_generic_to_shared(&counter);
    uint32_t old;
    asm volatile("atom.shared.add.u32 %0, [%1], 1;"
                 : "=r"(old) : "r"(saddr) : "memory");
    __syncthreads();

    if (threadIdx.x == 0) result[0] = counter;
}

// ── Test 5: Packed FP16 add — two FP16 additions in one instruction ─
// a=(a0,a1), b=(b0,b1) packed into .f16x2 registers.
// add.f16x2 adds both pairs simultaneously → r=(a0+b0, a1+b1).
__global__ void test_packed_fp16(uint32_t* out,
                                  float a0, float a1, float b0, float b1) {
    // Convert FP32→FP16 ("h" = 16-bit register)
    uint16_t ha0, ha1, hb0, hb1;
    asm("cvt.rn.f16.f32 %0,%1;" : "=h"(ha0) : "f"(a0));
    asm("cvt.rn.f16.f32 %0,%1;" : "=h"(ha1) : "f"(a1));
    asm("cvt.rn.f16.f32 %0,%1;" : "=h"(hb0) : "f"(b0));
    asm("cvt.rn.f16.f32 %0,%1;" : "=h"(hb1) : "f"(b1));

    // Pack: {lo16, hi16} → lo in bits[15:0], hi in bits[31:16]
    uint32_t pa, pb;
    asm("mov.b32 %0, {%1,%2};" : "=r"(pa) : "h"(ha0), "h"(ha1));
    asm("mov.b32 %0, {%1,%2};" : "=r"(pb) : "h"(hb0), "h"(hb1));

    // Packed add: ONE instruction handles TWO FP16 additions
    uint32_t pr;
    asm("add.f16x2 %0, %1, %2;" : "=r"(pr) : "r"(pa), "r"(pb));

    // Unpack and convert back to FP32 for verification
    uint16_t hr0, hr1;
    asm("mov.b32 {%0,%1}, %2;" : "=h"(hr0), "=h"(hr1) : "r"(pr));
    float r0, r1;
    asm("cvt.f32.f16 %0,%1;" : "=f"(r0) : "h"(hr0));
    asm("cvt.f32.f16 %0,%1;" : "=f"(r1) : "h"(hr1));

    out[0] = pr;
    out[1] = __float_as_uint(r0);
    out[2] = __float_as_uint(r1);
}

// ── Test 6: Bit reinterpret — mov.b32 float↔uint32 ───────────────
// mov.b32 moves raw bits with NO value conversion (unlike cvt).
// Useful for: reading IEEE 754 bit patterns, packing manipulations.
__global__ void test_bit_reinterpret(uint32_t* out) {
    float pi = 3.14159f;

    // float → uint32: get raw IEEE 754 bits (≈ 0x40490FD0)
    uint32_t bits;
    asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(pi));

    // uint32 → float: put bits back (reconstructed == pi exactly)
    float reconstructed;
    asm("mov.b32 %0, %1;" : "=f"(reconstructed) : "r"(bits));

    out[0] = bits;
    out[1] = __float_as_uint(reconstructed);
    out[2] = (bits == __float_as_uint(pi)) ? 1 : 0;  // should be 1
}

// ── Main ──────────────────────────────────────────────────────────
// Copy the representation without aliasing or device-only intrinsics.
static float host_uint_as_float(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  ex04: PTX Pointers, State Spaces & Packed Types      ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    int tests = 0, passed = 0;

    // Test 1
    printf("Test 1: cvta + ld.shared (smem[0]=42, smem[1]=99)\n");
    {
        uint32_t *d, h[3];
        CHECK_CUDA(cudaMalloc(&d, 3*4));
        test_cvta_shared<<<1,1>>>(d);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h,d,12,cudaMemcpyDeviceToHost));
        float v0=host_uint_as_float(h[0]), v1=host_uint_as_float(h[1]);
        tests++;
        if (fabsf(v0-42.0f)<1e-5f && fabsf(v1-99.0f)<1e-5f) {
            passed++;
            printf("  ✓ PASS: loaded %.1f and %.1f, SMEM offset=0x%X\n",v0,v1,h[2]);
        } else printf("  ✗ FAIL: got %.1f %.1f\n",v0,v1);
        cudaFree(d);
    }

    // Test 2
    printf("\nTest 2: st.shared/ld.shared neighbor (expect 7.0 and 17.0)\n");
    {
        float *d, h[2];
        CHECK_CUDA(cudaMalloc(&d, 8));
        test_explicit_shared<<<1,32>>>(d);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h,d,8,cudaMemcpyDeviceToHost));
        tests++;
        if (fabsf(h[0]-7.0f)<1e-5f && fabsf(h[1]-17.0f)<1e-5f) {
            passed++; printf("  ✓ PASS: my=%.1f neighbor=%.1f\n",h[0],h[1]);
        } else printf("  ✗ FAIL: my=%.1f neighbor=%.1f\n",h[0],h[1]);
        cudaFree(d);
    }

    // Test 3
    printf("\nTest 3: ld.global/st.global multiply by 2\n");
    {
        const int N=8;
        float hi[N]={1,2,3,4,5,6,7,8}, ho[N]={};
        float *di,*dout;
        CHECK_CUDA(cudaMalloc(&di,N*4)); CHECK_CUDA(cudaMalloc(&dout,N*4));
        CHECK_CUDA(cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice));
        test_global_ptx<<<1,N>>>(di,dout,N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(ho,dout,N*4,cudaMemcpyDeviceToHost));
        bool ok=true;
        for(int i=0;i<N;i++) if(fabsf(ho[i]-hi[i]*2)>1e-4f){ok=false;break;}
        tests++;
        if(ok){passed++; printf("  ✓ PASS: all elements×2 correct\n");}
        else printf("  ✗ FAIL\n");
        cudaFree(di); cudaFree(dout);
    }

    // Test 4
    printf("\nTest 4: atom.shared.add (64 threads → counter=64)\n");
    {
        uint32_t *d, h;
        CHECK_CUDA(cudaMalloc(&d,4));
        test_shared_atomic<<<1,64>>>(d);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost));
        tests++;
        if(h==64){passed++; printf("  ✓ PASS: counter=%u\n",h);}
        else printf("  ✗ FAIL: counter=%u (expected 64)\n",h);
        cudaFree(d);
    }

    // Test 5
    printf("\nTest 5: Packed FP16 add (1+3=4, 2+4=6)\n");
    {
        uint32_t *d, h[3];
        CHECK_CUDA(cudaMalloc(&d,12));
        test_packed_fp16<<<1,1>>>(d, 1.0f,2.0f, 3.0f,4.0f);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h,d,12,cudaMemcpyDeviceToHost));
        float r0=host_uint_as_float(h[1]), r1=host_uint_as_float(h[2]);
        tests++;
        if(fabsf(r0-4.0f)<0.01f && fabsf(r1-6.0f)<0.01f){
            passed++; printf("  ✓ PASS: slot0=%.1f slot1=%.1f\n",r0,r1);
        } else printf("  ✗ FAIL: slot0=%.4f slot1=%.4f\n",r0,r1);
        cudaFree(d);
    }

    // Test 6
    printf("\nTest 6: mov.b32 bit reinterpret (float↔uint32)\n");
    {
        uint32_t *d, h[3];
        CHECK_CUDA(cudaMalloc(&d,12));
        test_bit_reinterpret<<<1,1>>>(d);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h,d,12,cudaMemcpyDeviceToHost));
        tests++;
        if(h[2]==1){
            passed++;
            printf("  ✓ PASS: pi bits=0x%08X, roundtrip matches\n",h[0]);
        } else printf("  ✗ FAIL: mismatch 0x%08X vs 0x%08X\n",h[0],h[1]);
        cudaFree(d);
    }

    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  %d/%d tests passed %-34s║\n", passed, tests,
           passed==tests ? "  ✓ ALL PASS" : "  ✗ FAILURES");
    printf("╚══════════════════════════════════════════════════════╝\n");
    return (passed==tests) ? 0 : 1;
}
