// ═══════════════════════════════════════════════════════════════════
// ex02_ptx_basics.cu
//
// PURPOSE: Introduce inline PTX assembly from the ground up.
// Covers: integer arithmetic, FP arithmetic, FMA, predicates,
//         special registers, bitwise ops, scoped temporaries, templates.
//
// READING THIS CODE:
//   Each test kernel demonstrates ONE PTX concept.
//   Read the comment block before each kernel, then trace through
//   the "Manual trace" to understand what numbers flow where.
//   Then read the code to see how it's expressed in C++ with asm().
//
// WORKS ON: sm_80, sm_86, sm_89, sm_90a, sm_100a
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cmath>      // for fabsf, fabs

// ─── Error checking macro ──────────────────────────────────────────
#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)

// ─── Allocation helpers ────────────────────────────────────────────
// These macros reduce boilerplate for the common pattern of:
// allocate device memory → launch kernel → copy back → free.
#define ALLOC_DEV(ptr, count) \
    CHECK_CUDA(cudaMalloc(&ptr, (count) * sizeof(*ptr)))
#define FREE_DEV(ptr) \
    CHECK_CUDA(cudaFree(ptr))
#define COPY_FROM_DEV(host, dev, count) \
    CHECK_CUDA(cudaMemcpy(host, dev, (count)*sizeof(*host), cudaMemcpyDeviceToHost))

// ═══════════════════════════════════════════════════════════════════
// TEST 1: Integer Addition — add.u32
// ═══════════════════════════════════════════════════════════════════
//
// PTX instruction: add.u32 dst, src1, src2
//   Adds two unsigned 32-bit integers. Overflow wraps (no trap).
//   Equivalent to C++: dst = src1 + src2;
//
// Manual trace (a=42, b=10):
//   add.u32 %0, %1, %2
//   → result = 42 + 10 = 52
//
// Constraint breakdown:
//   "=r"(result) → %0 is result, write-only ("="), 32-bit int reg ("r")
//   "r"(a)       → %1 is a, read-only, 32-bit int reg
//   "r"(b)       → %2 is b, read-only, 32-bit int reg
__global__ void test_add_u32(uint32_t* out, uint32_t a, uint32_t b) {
    uint32_t result;
    asm("add.u32 %0, %1, %2;"
        : "=r"(result)      // %0 → result (write-only output)
        : "r"(a), "r"(b)    // %1 → a, %2 → b (read-only inputs)
    );
    out[0] = result;        // write to global memory for host to read
}

// ═══════════════════════════════════════════════════════════════════
// TEST 2: Multiply-Add (MAD) — mad.lo.u32
// ═══════════════════════════════════════════════════════════════════
//
// PTX instruction: mad.lo.u32 dst, a, b, c
//   Computes a×b+c, takes the LOW 32 bits (lo variant).
//   The full product a×b can be 64 bits; .lo discards the upper 32.
//   For inputs that fit in 32 bits, this is equivalent to a*b+c.
//
// Why MAD over separate MUL+ADD?
//   On GPU hardware, multiply and add are often fused at the
//   instruction level for better throughput (one instruction slot
//   instead of two). The compiler does this automatically for
//   integer code, but using MAD explicitly guarantees it.
//
// Manual trace (a=5, b=7, c=3):
//   mad.lo.u32 result, 5, 7, 3
//   → 5×7 = 35, 35+3 = 38
//   → result = 38
__global__ void test_mad_u32(uint32_t* out, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t result;
    asm("mad.lo.u32 %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(a), "r"(b), "r"(c)   // %1=a, %2=b, %3=c
    );
    out[0] = result;
}

// ═══════════════════════════════════════════════════════════════════
// TEST 3: Fused Multiply-Add (FMA) — fma.rn.f32
// ═══════════════════════════════════════════════════════════════════
//
// PTX instruction: fma.rn.f32 dst, a, b, c
//   Computes (a × b) + c with a SINGLE rounding step.
//
// Why FMA over separate mul then add?
//   mul.f32 a, b → rounds the intermediate product
//   add.f32 result, tmp, c → rounds again
//   Two roundings → less accurate result.
//
//   fma.rn.f32 → the product a×b is computed with infinite precision,
//   then c is added, then ONE rounding is applied at the end.
//   This is both more accurate AND more efficient (one instruction).
//
// .rn = round to nearest even (the IEEE 754 default mode)
// Other rounding modes: .rz (to zero), .rp (to +inf), .rm (to -inf)
//
// Manual trace (a=2.5, b=4.0, c=1.5):
//   result = (2.5 × 4.0) + 1.5 = 10.0 + 1.5 = 11.5 (exact, no rounding needed)
//
// Constraint: "f" means 32-bit floating-point register (.f32)
__global__ void test_fma_f32(float* out, float a, float b, float c) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(result)          // "f" = 32-bit floating-point register
        : "f"(a), "f"(b), "f"(c)
    );
    out[0] = result;
}

// ═══════════════════════════════════════════════════════════════════
// TEST 4: Predicate Registers — setp + selp
// ═══════════════════════════════════════════════════════════════════
//
// Predicates are 1-bit boolean registers in PTX. They are used for
// conditional operations without branches.
//
// Pattern: compare → set predicate → use predicate to select value
//
// Instructions:
//   setp.gt.s32  p, a, b    → p = (a > b)  [gt=greater-than, s32=signed]
//   selp.s32     dst, x, y, p → dst = p ? x : y  [ternary select]
//
// This pattern computes max(a,b) with NO branch instruction.
// The GPU executes both paths in SIMD style using predication.
//
// Manual trace (a=10, b=20):
//   p = (10 > 20) = false
//   result = false ? 10 : 20 = 20    ← correct max
//
// Manual trace (a=30, b=5):
//   p = (30 > 5) = true
//   result = true ? 30 : 5 = 30      ← correct max
//
// The { } block creates a PTX scope so ".reg .pred p" is local.
// Multiple asm() blocks can use the same name p without conflict.
__global__ void test_predicate_max(int32_t* out, int32_t a, int32_t b) {
    int32_t result;
    asm(
        "{\n\t"                              // open PTX scope
        "  .reg .pred p;\n\t"               // declare 1-bit predicate register
        "  setp.gt.s32 p, %1, %2;\n\t"     // p = (a > b)   [signed comparison]
        "  selp.s32 %0, %1, %2, p;\n\t"    // result = p ? a : b
        "}\n\t"                              // close PTX scope
        : "=r"(result)
        : "r"(a), "r"(b)
    );
    out[0] = result;
}

// ═══════════════════════════════════════════════════════════════════
// TEST 5: Special Registers — %tid.x, %ctaid.x, %ntid.x
// ═══════════════════════════════════════════════════════════════════
//
// PTX special registers give direct access to hardware registers
// that store the current thread's position in the execution hierarchy.
// They are read via "mov.u32 dst, %register_name;"
//
// Common special registers:
//   %tid.x, %tid.y, %tid.z    → threadIdx.x/y/z
//   %ntid.x, %ntid.y, %ntid.z → blockDim.x/y/z
//   %ctaid.x, %ctaid.y, %ctaid.z → blockIdx.x/y/z
//   %nctaid.x, ...            → gridDim.x/y/z
//   %laneid                   → lane within warp (0-31)
//   %warpid                   → warp within SM (0 to num_warps-1)
//   %smid                     → SM index (0 to num_SMs-1)
//   %clock                    → 32-bit cycle counter
//   %globaltimer               → 64-bit nanosecond timer
//
// Reading %tid.x via PTX is exactly equivalent to threadIdx.x in C++,
// but demonstrates that threadIdx is NOT a variable — it's a hardware
// register read that the CUDA compiler wraps in a friendly name.
__global__ void test_special_regs(uint32_t* tid_out, uint32_t* ctaid_out,
                                   uint32_t* laneid_out) {
    uint32_t tid, ctaid, laneid;
    asm("mov.u32 %0, %tid.x;"    : "=r"(tid));      // threadIdx.x
    asm("mov.u32 %0, %ctaid.x;"  : "=r"(ctaid));    // blockIdx.x
    asm("mov.u32 %0, %laneid;"   : "=r"(laneid));   // lane within warp

    // Only thread 0 of the first warp writes to avoid races
    if (tid == 0 && ctaid == 0) {
        tid_out[0]    = tid;
        ctaid_out[0]  = ctaid;
        laneid_out[0] = laneid;  // should be 0 since tid==0
    }
}

// ═══════════════════════════════════════════════════════════════════
// TEST 6: Bitwise Operations — and, or, xor, shl, shr
// ═══════════════════════════════════════════════════════════════════
//
// PTX bitwise instructions operate on .b32 (raw bits, no sign/type).
// They are identical to C++ bitwise operators but in PTX syntax.
//
// Manual trace (a=0xFF00, b=0x0F0F):
//   In binary:
//     a = 1111 1111 0000 0000
//     b = 0000 1111 0000 1111
//   and: 0000 1111 0000 0000 = 0x0F00
//   or:  1111 1111 0000 1111 = 0xFF0F
//   xor: 1111 0000 0000 1111 = 0xF00F
//   shl (a << 4):
//     a = 0000 0000 1111 1111 0000 0000 (32-bit)
//     a << 4 = 0000 1111 1111 0000 0000 0000 → low 32 bits = 0x00FF0000 = 0xFF000
//     Wait, let me redo: 0xFF00 = 65280 = 0b1111111100000000
//     << 4: 0b11111111000000000000 = 0xFF000 = 1044480 ✓
__global__ void test_bitwise(uint32_t* out, uint32_t a, uint32_t b) {
    uint32_t r_and, r_or, r_xor, r_shl, r_shr;

    asm("and.b32 %0, %1, %2;" : "=r"(r_and) : "r"(a), "r"(b));  // a & b
    asm("or.b32  %0, %1, %2;" : "=r"(r_or)  : "r"(a), "r"(b));  // a | b
    asm("xor.b32 %0, %1, %2;" : "=r"(r_xor) : "r"(a), "r"(b));  // a ^ b
    // Note: shift amount is a literal (4), not a variable.
    // PTX requires shift amounts to be immediate or in a register.
    asm("shl.b32 %0, %1, 4;"  : "=r"(r_shl) : "r"(a));           // a << 4
    asm("shr.u32 %0, %1, 4;"  : "=r"(r_shr) : "r"(a));           // a >> 4 (logical)
    // Note: shr.u32 is logical shift (fills with 0s).
    //       shr.s32 is arithmetic shift (fills with sign bit).

    out[0] = r_and;
    out[1] = r_or;
    out[2] = r_xor;
    out[3] = r_shl;
    out[4] = r_shr;
}

// ═══════════════════════════════════════════════════════════════════
// TEST 7: Multi-Instruction Block with Scoped PTX Temporaries
// ═══════════════════════════════════════════════════════════════════
//
// When you write multiple PTX instructions in a single asm() block,
// you can use temporary PTX registers with .reg declarations.
// The { } braces create a local scope for these declarations.
//
// Use documented NVIDIA constraints: "=r" for this output, "+r" for
// read/write operands. Both inputs are consumed into scoped PTX temporaries
// before the output is written. Do not infer physical register allocation
// from virtual PTX register names or require a GCC-only modifier.
//
// This kernel computes (a + b) * (a - b) = a² - b² using PTX temps.
//
// Manual trace (a=7, b=3):
//   sum  = 7 + 3 = 10
//   diff = 7 - 3 = 4
//   result = 10 × 4 = 40  (= 7² - 3² = 49 - 9 = 40 ✓)
__global__ void test_scoped_temporaries(int32_t* out, int32_t a, int32_t b) {
    int32_t result;
    asm(
        "{\n\t"
        "  .reg .s32 sum;\n\t"          // temporary register: sum
        "  .reg .s32 diff;\n\t"         // temporary register: diff
        "  add.s32 sum, %1, %2;\n\t"    // sum = a + b
        "  sub.s32 diff, %1, %2;\n\t"   // diff = a - b
        "  mul.lo.s32 %0, sum, diff;\n\t" // result = sum × diff
        "}\n\t"
        : "=r"(result)    // Output is written only after both inputs are consumed.
        : "r"(a), "r"(b)
    );
    out[0] = result;
}

// ═══════════════════════════════════════════════════════════════════
// TEST 8: Read-Write Operand — "+r" constraint
// ═══════════════════════════════════════════════════════════════════
//
// The "+" modifier means the operand is BOTH read and written.
// The same C++ variable is used as both input and output.
// This is useful for increment/decrement in-place operations.
//
// Manual trace (initial=100):
//   Step 1: add.u32 counter, counter, 1 → counter = 101
//   Step 2: add.u32 counter, counter, 1 → counter = 102
//   Step 3: add.u32 counter, counter, 1 → counter = 103
//
// The "+r"(counter) tells the compiler:
//   - counter occupies a 32-bit register (%0)
//   - before the instruction: %0 = counter (READ from counter)
//   - after the instruction: counter = %0 (WRITE to counter)
//   - %0 appears on BOTH sides of the instruction
__global__ void test_readwrite(uint32_t* out, uint32_t initial) {
    uint32_t counter = initial;   // start value

    // Each of these reads counter, adds 1, writes back.
    // The "+r" constraint makes %0 appear on both left and right.
    asm("add.u32 %0, %0, 1;" : "+r"(counter));   // 100 → 101
    asm("add.u32 %0, %0, 1;" : "+r"(counter));   // 101 → 102
    asm("add.u32 %0, %0, 1;" : "+r"(counter));   // 102 → 103

    out[0] = counter;
}

// ═══════════════════════════════════════════════════════════════════
// TEST 9: Immediate Operands — "n" constraint with templates
// ═══════════════════════════════════════════════════════════════════
//
// Some PTX instructions require COMPILE-TIME constant operands.
// The "n" constraint tells the compiler that the operand must be
// a compile-time integer constant (not a runtime variable).
//
// Templates are the natural C++ mechanism for compile-time constants.
// The template parameter SHIFT_AMOUNT is known at compile time, so
// when the template is instantiated, the compiler can substitute
// the literal value directly into the PTX instruction string.
//
// This is exactly how CUTLASS uses template parameters for tile sizes.
//
// Manual trace (val=1, SHIFT_AMOUNT=5):
//   shl.b32 result, 1, 5
//   → 1 << 5 = 32
//
// Manual trace (val=7, SHIFT_AMOUNT=3):
//   shl.b32 result, 7, 3
//   → 7 << 3 = 56
template<int SHIFT_AMOUNT>
__global__ void test_immediate(uint32_t* out, uint32_t val) {
    uint32_t result;
    asm("shl.b32 %0, %1, %2;"
        : "=r"(result)
        : "r"(val),
          "n"(SHIFT_AMOUNT)   // "n" = immediate constant (compile-time only!)
                              // SHIFT_AMOUNT must be known at compile time.
                              // If you try "n"(runtime_var), it will fail to compile.
    );
    out[0] = result;
}

// ═══════════════════════════════════════════════════════════════════
// TEST 10: Volatile and Memory Clobber
// ═══════════════════════════════════════════════════════════════════
//
// WITHOUT volatile: the compiler may reorder, merge, or delete asm().
// WITH volatile: the compiler MUST execute it exactly where written.
//
// The "memory" clobber tells the compiler:
//   "This instruction may access memory you don't know about.
//    Invalidate all cached register copies of memory after this."
//
// Use volatile + "memory" for:
//   - Stores (side effects that must happen)
//   - Atomics (must happen in order)
//   - Barriers (ordering constraints)
//   - Any instruction that reads/writes memory implicitly
//
// This test writes to shared memory via PTX store (asm volatile with "memory")
// then reads back to verify the store actually happened.
__global__ void test_volatile_memory(uint32_t* out) {
    __shared__ uint32_t smem_val;

    if (threadIdx.x == 0) {
        // Step 1: Get shared memory address
        uint32_t smem_addr = __cvta_generic_to_shared(&smem_val);

        // Step 2: Store 0xDEADBEEF via volatile PTX store
        // "volatile" = don't reorder this store
        // "memory"   = this store touches memory the compiler can't see
        uint32_t value = 0xDEADBEEF;
        asm volatile("st.shared.b32 [%0], %1;"
                     :                           // no output (store has no "return value")
                     : "r"(smem_addr), "r"(value)
                     : "memory"                  // MEMORY CLOBBER: forces re-read later
        );

        // Step 3: Read back via PTX load
        // Because of "memory" clobber above, compiler won't cache
        // the old value of smem_val in a register. It must re-read.
        uint32_t loaded;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(loaded)
                     : "r"(smem_addr)
        );

        out[0] = loaded;  // should be 0xDEADBEEF
    }
}

// ═══════════════════════════════════════════════════════════════════
// TEST 11: Warp-Level Intrinsic via PTX — __ballot_sync equivalent
// ═══════════════════════════════════════════════════════════════════
//
// vote.sync.ballot.b32 computes a 32-bit mask where bit i is set if
// thread i in the warp has the input predicate set.
// This is exactly what __ballot_sync(mask, predicate) does in C++.
//
// Showing the PTX version demonstrates that CUDA intrinsics are
// thin wrappers around PTX instructions.
//
// Manual trace (all 32 threads have predicate = true):
//   vote.sync.ballot.b32 result, true, 0xffffffff
//   → result = 0xFFFFFFFF (all 32 bits set)
//
// If only even threads (0,2,4,...30) have predicate=true:
//   → result = 0x55555555 (bits 0,2,4,...30 set)
__global__ void test_ballot(uint32_t* out) {
    // Each thread checks: is my lane ID even?
    uint32_t lane;
    asm("mov.u32 %0, %laneid;" : "=r"(lane));
    uint32_t predicate = (lane % 2 == 0) ? 1 : 0;

    uint32_t ballot_result;
    asm volatile("{ .reg .pred p; setp.ne.u32 p, %1, 0; "
                 "vote.sync.ballot.b32 %0, p, 0xffffffff; }"
        : "=r"(ballot_result)
        : "r"(predicate)
    );
    // ballot_result: bit i set iff thread i had predicate=1
    // Since even threads have predicate=1: 0x55555555

    if (lane == 0) {
        out[0] = ballot_result;
    }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN — Run all tests and print results
// ═══════════════════════════════════════════════════════════════════
int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Inline PTX Basics — Full Test Suite                  ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    int tests = 0, passed = 0;

    // ─── Test 1: add.u32 ─────────────────────────────────────────
    // Expected: 42 + 10 = 52
    printf("Test 1: add.u32 (42 + 10 = 52)\n");
    {
        uint32_t *d_out, h_out;
        ALLOC_DEV(d_out, 1);
        test_add_u32<<<1, 1>>>(d_out, 42, 10);
        CHECK_CUDA(cudaDeviceSynchronize());
        COPY_FROM_DEV(&h_out, d_out, 1);
        tests++;
        if (h_out == 52) { passed++; printf("  ✓ PASS: result = %u\n", h_out); }
        else printf("  ✗ FAIL: got %u, expected 52\n", h_out);
        FREE_DEV(d_out);
    }

    // ─── Test 2: mad.lo.u32 ──────────────────────────────────────
    // Expected: 5*7 + 3 = 38
    printf("\nTest 2: mad.lo.u32 (5*7+3 = 38)\n");
    {
        uint32_t *d_out, h_out;
        ALLOC_DEV(d_out, 1);
        test_mad_u32<<<1, 1>>>(d_out, 5, 7, 3);
        CHECK_CUDA(cudaDeviceSynchronize());
        COPY_FROM_DEV(&h_out, d_out, 1);
        tests++;
        if (h_out == 38) { passed++; printf("  ✓ PASS: result = %u\n", h_out); }
        else printf("  ✗ FAIL: got %u, expected 38\n", h_out);
        FREE_DEV(d_out);
    }

    // ─── Test 3: fma.rn.f32 ──────────────────────────────────────
    // Expected: 2.5 * 4.0 + 1.5 = 11.5
    printf("\nTest 3: fma.rn.f32 (2.5 * 4.0 + 1.5 = 11.5)\n");
    {
        float *d_out, h_out;
        ALLOC_DEV(d_out, 1);
        test_fma_f32<<<1, 1>>>(d_out, 2.5f, 4.0f, 1.5f);
        CHECK_CUDA(cudaDeviceSynchronize());
        COPY_FROM_DEV(&h_out, d_out, 1);
        tests++;
        if (fabsf(h_out - 11.5f) < 1e-5f) {
            passed++; printf("  ✓ PASS: result = %.4f\n", h_out);
        } else printf("  ✗ FAIL: got %.4f, expected 11.5\n", h_out);
        FREE_DEV(d_out);
    }

    // ─── Test 4: Predicate max ───────────────────────────────────
    printf("\nTest 4: predicate max\n");
    {
        int32_t *d_out, h_out;
        ALLOC_DEV(d_out, 1);

        // max(10, 20) should be 20
        test_predicate_max<<<1, 1>>>(d_out, 10, 20);
        CHECK_CUDA(cudaDeviceSynchronize());
        COPY_FROM_DEV(&h_out, d_out, 1);
        tests++;
        if (h_out == 20) { passed++; printf("  ✓ PASS: max(10,20) = %d\n", h_out); }
        else printf("  ✗ FAIL: got %d, expected 20\n", h_out);

        // max(30, 5) should be 30
        test_predicate_max<<<1, 1>>>(d_out, 30, 5);
        CHECK_CUDA(cudaDeviceSynchronize());
        COPY_FROM_DEV(&h_out, d_out, 1);
        tests++;
        if (h_out == 30) { passed++; printf("  ✓ PASS: max(30,5) = %d\n", h_out); }
        else printf("  ✗ FAIL: got %d, expected 30\n", h_out);

        FREE_DEV(d_out);
    }

    // ─── Test 5: Special registers ───────────────────────────────
    printf("\nTest 5: special registers (%%tid.x, %%ctaid.x, %%laneid)\n");
    {
        uint32_t *d_tid, *d_ctaid, *d_lane;
        uint32_t h_tid, h_ctaid, h_lane;
        ALLOC_DEV(d_tid, 1);
        ALLOC_DEV(d_ctaid, 1);
        ALLOC_DEV(d_lane, 1);

        // Launch 2 blocks of 64 threads each.
        // Block 0, thread 0, lane 0 will write the output.
        test_special_regs<<<2, 64>>>(d_tid, d_ctaid, d_lane);
        CHECK_CUDA(cudaDeviceSynchronize());
        COPY_FROM_DEV(&h_tid,   d_tid,   1);
        COPY_FROM_DEV(&h_ctaid, d_ctaid, 1);
        COPY_FROM_DEV(&h_lane,  d_lane,  1);

        // Thread 0 of block 0 writes: tid=0, ctaid=0, laneid=0
        tests++;
        if (h_tid == 0) { passed++; printf("  ✓ PASS: tid.x = %u\n", h_tid); }
        else printf("  ✗ FAIL: tid.x = %u (expected 0)\n", h_tid);

        printf("  INFO: ctaid.x = %u (block index, not checked — scheduling dependent)\n",
               h_ctaid);
        printf("  INFO: laneid = %u (should be 0 since tid=0)\n", h_lane);

        FREE_DEV(d_tid); FREE_DEV(d_ctaid); FREE_DEV(d_lane);
    }

    // ─── Test 6: Bitwise operations ──────────────────────────────
    // a=0xFF00, b=0x0F0F
    // Expected: and=0x0F00, or=0xFF0F, xor=0xF00F, shl=0xFF000, shr=0x0FF0
    printf("\nTest 6: bitwise ops (a=0xFF00, b=0x0F0F)\n");
    {
        uint32_t *d_out, h_out[5];
        ALLOC_DEV(d_out, 5);
        test_bitwise<<<1, 1>>>(d_out, 0xFF00, 0x0F0F);
        CHECK_CUDA(cudaDeviceSynchronize());
        COPY_FROM_DEV(h_out, d_out, 5);

        struct { const char* name; uint32_t got; uint32_t exp; } checks[] = {
            {"and", h_out[0], 0x0F00},
            {"or",  h_out[1], 0xFF0F},
            {"xor", h_out[2], 0xF00F},
            {"shl", h_out[3], 0xFF000},
            {"shr", h_out[4], 0x0FF0},  // 0xFF00 >> 4 = 0x0FF0
        };
        for (auto& c : checks) {
            tests++;
            if (c.got == c.exp) {
                passed++; printf("  ✓ PASS: %s = 0x%05X\n", c.name, c.got);
            } else {
                printf("  ✗ FAIL: %s = 0x%X, expected 0x%X\n", c.name, c.got, c.exp);
            }
        }
        FREE_DEV(d_out);
    }

    // ─── Test 7: Early clobber ────────────────────────────────────
    // Expected: (7+3) * (7-3) = 10 * 4 = 40
    printf("\nTest 7: scoped temporaries — (7+3)*(7-3) = 40\n");
    {
        int32_t *d_out, h_out;
        ALLOC_DEV(d_out, 1);
        test_scoped_temporaries<<<1, 1>>>(d_out, 7, 3);
        CHECK_CUDA(cudaDeviceSynchronize());
        COPY_FROM_DEV(&h_out, d_out, 1);
        tests++;
        if (h_out == 40) { passed++; printf("  ✓ PASS: result = %d\n", h_out); }
        else printf("  ✗ FAIL: got %d, expected 40\n", h_out);
        FREE_DEV(d_out);
    }

    // ─── Test 8: Read-write operand ───────────────────────────────
    // Expected: 100 + 3 = 103 (three in-place increments)
    printf("\nTest 8: read-write operand — 100 + 3 = 103\n");
    {
        uint32_t *d_out, h_out;
        ALLOC_DEV(d_out, 1);
        test_readwrite<<<1, 1>>>(d_out, 100);
        CHECK_CUDA(cudaDeviceSynchronize());
        COPY_FROM_DEV(&h_out, d_out, 1);
        tests++;
        if (h_out == 103) { passed++; printf("  ✓ PASS: result = %u\n", h_out); }
        else printf("  ✗ FAIL: got %u, expected 103\n", h_out);
        FREE_DEV(d_out);
    }

    // ─── Test 9: Immediate operand (template) ─────────────────────
    printf("\nTest 9: immediate operand template\n");
    {
        uint32_t *d_out, h_out;
        ALLOC_DEV(d_out, 1);

        // 1 << 5 = 32
        test_immediate<5><<<1, 1>>>(d_out, 1);
        CHECK_CUDA(cudaDeviceSynchronize());
        COPY_FROM_DEV(&h_out, d_out, 1);
        tests++;
        if (h_out == 32) { passed++; printf("  ✓ PASS: 1 << 5 = %u\n", h_out); }
        else printf("  ✗ FAIL: got %u, expected 32\n", h_out);

        // 7 << 3 = 56
        test_immediate<3><<<1, 1>>>(d_out, 7);
        CHECK_CUDA(cudaDeviceSynchronize());
        COPY_FROM_DEV(&h_out, d_out, 1);
        tests++;
        if (h_out == 56) { passed++; printf("  ✓ PASS: 7 << 3 = %u\n", h_out); }
        else printf("  ✗ FAIL: got %u, expected 56\n", h_out);

        FREE_DEV(d_out);
    }

    // ─── Test 10: Volatile + memory clobber ───────────────────────
    printf("\nTest 10: volatile + memory clobber (store 0xDEADBEEF to SMEM)\n");
    {
        uint32_t *d_out, h_out;
        ALLOC_DEV(d_out, 1);
        test_volatile_memory<<<1, 32>>>(d_out);
        CHECK_CUDA(cudaDeviceSynchronize());
        COPY_FROM_DEV(&h_out, d_out, 1);
        tests++;
        if (h_out == 0xDEADBEEF) {
            passed++; printf("  ✓ PASS: loaded 0x%08X from SMEM\n", h_out);
        } else printf("  ✗ FAIL: got 0x%08X, expected 0xDEADBEEF\n", h_out);
        FREE_DEV(d_out);
    }

    // ─── Test 11: Ballot ──────────────────────────────────────────
    // Even lanes (0,2,4,...,30) have predicate=1 → ballot = 0x55555555
    printf("\nTest 11: vote.ballot (even lanes → 0x55555555)\n");
    {
        uint32_t *d_out, h_out;
        ALLOC_DEV(d_out, 1);
        test_ballot<<<1, 32>>>(d_out);
        CHECK_CUDA(cudaDeviceSynchronize());
        COPY_FROM_DEV(&h_out, d_out, 1);
        tests++;
        if (h_out == 0x55555555) {
            passed++; printf("  ✓ PASS: ballot = 0x%08X\n", h_out);
        } else printf("  ✗ FAIL: got 0x%08X, expected 0x55555555\n", h_out);
        FREE_DEV(d_out);
    }

    // ─── Final Summary ────────────────────────────────────────────
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  Results: %2d/%2d tests passed                          ║\n",
           passed, tests);
    if (passed == tests)
        printf("║  ✓ ALL PASS                                          ║\n");
    else
        printf("║  ✗ SOME FAILURES — see output above                  ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    return (passed == tests) ? 0 : 1;
}
