# 30. CUTLASS Attention Foundations — Work Every Softmax State
## Lesson 30 Companion Guide — From Two GEMMs to One Streaming Attention Contract

[![Chapter 30: CUTLASS attention foundations](images/chapter-openers/30-workshop.svg)](images/chapter-openers/30-workshop.svg)

!!! tip "Your goal in this chapter"

    Compute a complete attention example before reading a fused kernel. Track its scores, mask, row maximum, denominator and weighted-value numerator; derive its layouts independently of its arithmetic. Read [FlashAttention-3](12-flashattention-3.md), [strides](16-cute-strides.md), [the GEMM adapter](24-cutlass-gemm.md) and [pipeline ownership](25-cutlass-pipelines.md) first. [Chapter 31](31-cutlass-hopper-fmha.md) connects this contract to pinned Hopper FMHA code.

The CUTLASS example studied here is **`examples/88_hopper_fmha` in v4.3.6**, commit **`1afc6d355b8cb992398f17beb4425702f1d57ddd`**. Its README describes a forward GEMM–online-softmax–GEMM fusion and separate backward kernels. This chapter supplies the small mathematical bridge; it does not claim that the tiny example is a supported GPU launch or that a new GPU execution occurred.

The accompanying [complete CPU oracle](examples/advanced/attention_walkthrough.py) uses Python binary64 arithmetic and the standard library. It independently compares a materialized reference with streaming updates. That checks the worked equations, masks and state transitions. It does not validate CUDA instruction ordering, FP16/FP8 rounding, the CUTLASS compiler interface or performance.

## 1 — Define the operation and every axis

For one batch and one attention head, write

`S = (Q K^T) / sqrt(d)`

`P[i,j] = exp(S[i,j]) / sum_t exp(S[i,t])`

`O = P V`.

| Symbol | Shape | Meaning |
| --- | --- | --- |
| Q | Nq × d | One feature row per query |
| K | Nk × d | One feature row per key |
| V | Nk × dv | One output-value row per key |
| S, P | Nq × Nk | Scores and row-normalized probabilities |
| O | Nq × dv | Weighted value result per query |

The reduction in `Q K^T` is over d. The reduction in `P V` is over Nk. Confusing those axes gives a plausible-shaped intermediate that computes another operation. The mathematical transpose `K^T` does not require physically copying K into a transposed allocation; a compatible tensor view/instruction layout can represent the needed access.

This mathematical chapter permits `dv` distinct from d. The pinned example runner instead uses one D for Q, K, V and O. Likewise, no dropout, bias, rotary-position transformation or output projection is present in the toy contract. Additions need an explicit placement and numerical reference; they are not implied by the word attention.

## 2 — Choose inputs whose softmax can be checked exactly on paper

Let `a=sqrt(2)*ln(2)`, choose `d=dv=2`, and use:

```text
Q = [ a  0 ]      K = [ 1  0 ]      V = [ 1  0 ]
    [ 0  a ]          [ 0  1 ]          [ 0  2 ]
                      [ 1  1 ]          [ 3  4 ]
```

Here `Nq=2`, `Nk=3`. The numbers are a mathematical construction, not exactly representable FP16 inputs. Python approximates the transcendental constants in binary64, so its checks use an explicit `1e-12` relative/absolute tolerance. The hand derivation treats the equations over real numbers.

1. Query row zero dotted with key zero gives a; key one gives zero; key two gives a.
2. Scaling by `1/sqrt(2)` yields score row zero `[ln(2),0,ln(2)]`.
3. Query row one produces `[0,ln(2),ln(2)]`.
4. The unshifted exponentials are `[2,1,2]` and `[1,2,2]`. Each row sums to five.

Therefore the noncausal probability rows are `[2/5,1/5,2/5]` and `[1/5,2/5,2/5]`. Each has sum one and nonnegative entries. A sum-one check is necessary but cannot detect every key permutation; the nonuniform V values make that permutation observable at the output.

## 3 — Finish the second GEMM by hand

For output row zero, weight all three V rows:

`(2/5)*[1,0] + (1/5)*[0,2] + (2/5)*[3,4] = [8/5,2]`.

For output row one:

`(1/5)*[1,0] + (2/5)*[0,2] + (2/5)*[3,4] = [7/5,12/5]`.

The expected matrix is thus `O=[[1.6,2],[1.4,2.4]]`. The first component of row zero includes `2/5+6/5=8/5`; the second includes `2/5+8/5=2`. These are different weighted sums, not copies of the same softmax denominator.

**Pitfall:** multiplying every score row by V before normalizing would give unnormalized numerators `[8,10]` and `[7,12]`. Dividing by five is essential. Another wrong shortcut is normalizing each key tile independently and adding its outputs; tile-local denominators do not equal the complete row denominator.

## 4 — Use a stable maximum without changing the function

Subtracting one common value from a row of logits leaves its softmax unchanged. Choose row maximum m:

`exp(s_j)/sum_t exp(s_t) = exp(s_j-m)/sum_t exp(s_t-m)`.

For row zero, `m=ln(2)`. The stable exponential weights are `[1,1/2,1]`, denominator `l=5/2`, and weighted-value numerator `z=[4,5]`. Then `z/l=[8/5,2]` as before.

The denominator l is a scalar; z is a vector of length dv. Their magnitudes depend on the chosen maximum, but the normalized output does not. Save `LSE=m+ln(l)` if the implementation's contract requires natural-log log-sum-exp. Both noncausal rows have `LSE=ln(5)` in this example.

Some kernels evaluate exponentials with base two. The equivalent expression is `exp(s)=exp2(s*log2(e))`; it changes the scaling constant and the meaning of an internal maximum, not the requested function. Never mix a base-two maximum with a natural-exponential rescaling formula. Consult the selected implementation's units before comparing internal traces.

## 5 — Derive the online recurrence, including the numerator

A fused forward pass need not store the full P or S matrix in global memory. For one query row, retain `(m,l,z)` for all keys processed so far. Initially `m=-infinity`, `l=0`, `z=0`. For a new block of valid keys J:

`m_new = max(m, max_{j in J} s_j)`

`alpha = exp(m-m_new)`

`l_new = alpha*l + sum_{j in J} exp(s_j-m_new)`

`z_new = alpha*z + sum_{j in J} exp(s_j-m_new)*V[j,:]`.

1. Every old contribution was scaled relative to m.
2. The new state is scaled relative to `m_new`.
3. Multiplying the old denominator **and numerator** by alpha converts them to that new reference.
4. New contributions can then be added under one scale.
5. After the final actual key block, divide z by l once to produce O.

The first nonempty block uses alpha zero explicitly in the oracle, avoiding an undefined `-infinity-(-infinity)` calculation. A block containing no valid keys is skipped. An entirely masked row is rejected by this oracle rather than inventing a universal GPU convention for zero outputs or LSE.

## 6 — Follow every state through two real key blocks

[![Tiny Q/K/V scores, noncausal and causal results, and every online-softmax state](images/worked/ch30-online-softmax.svg)](images/worked/ch30-online-softmax.svg)

Split keys into block zero `{0,1}` and block one `{2}`. The second block is shorter; its third key is real, while nonexistent padding keys must not gain softmax weight.

| Query row / block | m | alpha for old state | l | z |
| --- | --- | ---: | ---: | --- |
| Row 0, keys 0–1 | ln(2) | 0 | 1.5 | [1,1] |
| Row 0, key 2 | ln(2) | 1 | 2.5 | [4,5] |
| Row 1, keys 0–1 | ln(2) | 0 | 1.5 | [0.5,2] |
| Row 1, key 2 | ln(2) | 1 | 2.5 | [3.5,6] |

For row zero's first block, stable weights one and one-half give `z=1*[1,0]+0.5*[0,2]=[1,1]`. Key two contributes `[3,4]` at stable weight one, so the final numerator is `[4,5]`. Divide by 2.5 to obtain `[1.6,2]`.

For row one, the first two stable weights are one-half and one. Its first numerator is `[0.5,2]`; adding key two yields `[3.5,6]`. Dividing by the same denominator gives `[1.4,2.4]`. Equal denominators do not make the two output vectors equal.

**Check:** the sequential trace and the materialized full-row reference agree. The CPU oracle also checks different valid partitions and traversal orders. Floating-point results need not be bit-identical under reassociation; matching within this stated tolerance is the criterion for these binary64 inputs.

## 7 — Force a maximum change so rescaling cannot be omitted

The previous order never increased m after the first tile. That cannot expose a missing rescale. Process row zero's key one first, then keys zero and two:

1. Key one has score zero and V row `[0,2]`. State is `m=0,l=1,z=[0,2]`.
2. The new block has scores `ln(2),ln(2)`, so `m_new=ln(2)`.
3. Old-state factor is `alpha=exp(-ln(2))=1/2`.
4. Rescale old `l` to 0.5 and old z to `[0,1]`.
5. Add two new weights of one: `l_new=0.5+1+1=2.5`.
6. Add new weighted vectors `[1,0]+[3,4]` to `[0,1]`, producing `[4,5]`.
7. Final output remains `[1.6,2]`.

If only l is rescaled, z incorrectly becomes `[4,6]`, giving `[1.6,2.4]`. If neither is rescaled, l becomes three and z `[4,6]`, giving `[4/3,2]`. A correct-looking component or normalized row is insufficient; check the whole vector against the independent reference.

## 8 — Define causal positions before applying a mask

The pinned `CausalFusion::before_softmax` demonstrates **top-left causal alignment**: a key is excluded when its logical index j exceeds query index i. For `Nq=2,Nk=3`, the Boolean mask is:

```text
                key j=0  j=1  j=2
query i=0          yes    no   no
query i=1          yes   yes   no
```

1. Row zero retains only key zero, so its normalized weight is one and output is `[1,0]`.
2. Row one retains scores `[0,ln(2)]`, whose exponentials are `[1,2]`.
3. Its weights are `[1/3,2/3,0]` and output is `[1/3,4/3]`.
4. The corresponding natural-log LSE values are `ln(2)` and `ln(3)`.

Mask logits before softmax: an excluded entry contributes zero weight and does not enter the row maximum or denominator. Setting an excluded logit to zero is wrong because `exp(0)=1`. The CPU helper represents masking with a separate Boolean `allowed` list while retaining finite logits; it does not accept literal infinity/NaN score inputs. Thus the mathematical negative-infinity mask and the helper's input representation are different. An entirely masked block is skipped, and an entirely masked row is rejected under its stated contract. Zero-filling an out-of-range K tile similarly does not automatically mask its score; it can create a false zero logit with positive softmax mass.

Rectangular attention makes the position convention visible. If the two queries represent the *last* two positions of a three-key sequence, use absolute query positions `i+1`. The valid test becomes `j<=i+1`, and the outputs are `[[2/3,2/3],[1.4,2.4]]`. This is a separate oracle contract, not what the pinned causal functor silently does. Its own comments explain where such an offset would be introduced. Head sharing, cached decode and unequal lengths require explicit positions and ownership, not merely `causal=true`.

## 9 — Separate memory offsets from score coordinates

A score coordinate `(i,j)` names a query/key pair, whereas a Q/K element coordinate names one feature. For compact row-major views use:

`Q(i,c): i*d+c`, `K(j,c): j*d+c`, `V(j,v): j*dv+v`, `O(i,v): i*dv+v`.

| View coordinate | Element offset | FP16 byte offset, if E=2 |
| --- | ---: | ---: |
| Q(1,1) | 1*2+1=3 | 6 |
| K(2,1) | 2*2+1=5 | 10 |
| V(2,0) | 2*2+0=4 | 8 |
| O(1,0) | 1*2+0=2 | 4 |

This byte column illustrates a hypothetical FP16 allocation; it does not claim that the CPU oracle used half storage or reproduced half rounding. With eight-byte binary64 elements, multiply the same offsets by eight instead. K's mathematical transpose in QKᵀ does not change its row-major `(j,c)` storage formula.

For two batches and three heads, choose illustrative mode order `(sequence,feature,(batch,head))` with Q strides `(d,1,(H*Nq*d,Nq*d))`. At `(i,c,b,h)=(1,1,1,2)`, offset is `1*2+1+1*12+2*4=23` elements, or 46 bytes for FP16. The last valid element occupies bytes 46–47 of the 48-byte Q allocation. K/V use Nk instead of Nq, so their batch/head strides are different.

The pinned sixteen-bit runner uses the same **mode order**, with `StrideQ`/`StrideK` represented as `tuple<int,_1,tuple<int,int>>`. Its FP8 V specialization uses a different stride pattern; do not transplant the sixteen-bit V map into the byte-element path. Supported TMA alignment and head dimensions impose additional restrictions beyond this affine address calculation.

## 10 — Map the mathematics to the pinned implementation

The [pinned example README](https://github.com/NVIDIA/cutlass/blob/1afc6d355b8cb992398f17beb4425702f1d57ddd/examples/88_hopper_fmha/README.md) describes four layers:

| Layer | What to locate | Mathematical obligation |
| --- | --- | --- |
| Runner/reference | Inputs, strides, checks and timing | Same operation and numerical criterion |
| Device operation | Workspace and launch orchestration | Correct pointers, shapes, lifetimes and completion |
| Kernel | Scheduling and roles | Complete query/output coverage |
| Collectives | QK, softmax state, PV and epilogue | Correct reductions, rescaling and output |

A small type excerpt from the [actual forward runner](https://github.com/NVIDIA/cutlass/blob/1afc6d355b8cb992398f17beb4425702f1d57ddd/examples/88_hopper_fmha/88_hopper_fmha.cu) exposes the construction boundary:

```cpp
// Excerpt: aliases/headers/template parameters come from the complete runner.
using Operation = cutlass::device::Universal<
    typename cutlass::fmha::kernel::FmhaBuilder<
      Element, ElementAccumulatorQK, ElementAccumulatorPV,
      TileShape, StrideQ, StrideK, StrideV,
      ActiveFusion, DispatchPolicy, KernelOptions...
    >::Kernel>;
```

This is not an ordinary GEMM builder with a different activation tag. Softmax is between two matrix products, and both row statistics and output numerators persist across key tiles. The implementation uses FMHA-specific collective/kernel/device components. The runner's problem tuple is `(B,H,Q,K,D)`; Q and K in that tuple are **sequence lengths**, not tensor pointers. Compare the tuple with the data and stride arguments before constructing a launch.

The pinned README lists forward FP16/BF16 head dimensions 32,64,128,256 and FP8 dimensions 128,256; backward is separately FP16 with dimensions 32,64,128. These are advertised example paths, not proof of all arbitrary shapes, masks and strides. The toy `d=2` belongs only to this mathematical chapter. A successful address calculation is not a supported launch.

The customization hook `before_softmax` receives score accumulators and their logical positions. `ResidualFusion` masks keys beyond Nk; `CausalFusion` applies the demonstrated causal comparison. Inspect the selected fusion's complete behavior rather than assuming every mask automatically combines every kind of tail. A real ragged causal configuration must be checked against its exact specialization and reference.

## 11 — Keep intermediate lifetime and arithmetic separate

A fused implementation still needs Q/K/V delivery, safe buffer reuse, complete WGMMA participation and valid output stores. The online recurrence proves how to combine mathematical contributions; it does not establish TMA arrival or asynchronous-reader completion. Conversely, correct barriers do not repair a wrong rescaling formula.

Map one physical stage to its logical key interval. Record when K/V become ready, which query rows consume them, when asynchronous readers finish, and which producer may next overwrite the stage. A probability fragment may be converted before PV; that conversion changes numerical error even if the state recurrence is algebraically correct. Match the reference to the selected accumulator/conversion policy.

Avoid claiming that streaming attention reads each input from HBM exactly once. Query tiling, cache residency and schedule affect requested and physical traffic. The important storage difference is that a full Nq×Nk score/probability matrix need not be materialized in global memory. Benchmark only after specifying dtype, mask, lengths, heads, position alignment, work count and useful-result boundary.

## 12 — Run the CPU oracle and read its evidence

From the repository root:

```bash
python3 docs/examples/advanced/attention_walkthrough.py
```

The oracle checks materialized versus online output and natural-log LSE; noncausal and two explicitly positioned causal masks; a maximum-increasing rescale; several partitions/traversal orders; a +1000 common logit shift; an initially fully masked block; duplicate/missing coverage rejection; and an all-masked-row diagnostic. It uses explicit checks that remain active under Python optimization.

Its output is CPU mathematical evidence only. Compile/run commands, UTC times, local platform and exact source hash appear in the capture below. CUTLASS/Hopper compilation, vendor numerical checks and GPU profiling are a separate task in [Chapter 31](31-cutlass-hopper-fmha.md).

<!-- attention-walkthrough:capture:start -->
**Actually executed locally on 2026-11-18T21:34:25.783435+00:00 (UTC).** Scope: CPU-only binary64 mathematical reference. System: `macOS-26.6-arm64-arm-64bit-Mach-O`; Python `3.14.5`. Python byte-compilation and execution both exited zero.

Source SHA-256: `2e59d020aaf1c7d763bff8f7a924ff20da3b34f578f07ea5735024475367ea52`. [Exact captured source](validation/2026-11-18-attention-cpu/attention_walkthrough.py) · [System information](validation/2026-11-18-attention-cpu/system.json) · [Commands and stage times](validation/2026-11-18-attention-cpu/commands.json) · [Compile log](validation/2026-11-18-attention-cpu/compile.log) · [Run log](validation/2026-11-18-attention-cpu/run.log).

The raw commands ran in the capture directory with its copied source. Equivalent scratch-directory commands are:

```bash
python3 -m py_compile attention_walkthrough.py
python3 attention_walkthrough.py
```

**Actual captured output:**

```text
NONCAUSAL O: [1.6, 2]; [1.4, 2.4]
ROW 0 BLOCK 0 keys=[0, 1]: m=0.69314718056 alpha=0 l=1.5 z=[1, 1]
ROW 0 BLOCK 1 keys=[2]: m=0.69314718056 alpha=1 l=2.5 z=[4, 5]
ROW 1 BLOCK 0 keys=[0, 1]: m=0.69314718056 alpha=0 l=1.5 z=[0.5, 2]
ROW 1 BLOCK 1 keys=[2]: m=0.69314718056 alpha=1 l=2.5 z=[3.5, 6]
TOP-LEFT CAUSAL O: [1, 0]; [0.333333333333, 1.33333333333]
OFFSET-1 CAUSAL O: [0.666666666667, 0.666666666667]; [1.4, 2.4]
RESCALE row0: alpha=0.5 l=2.5 z=[4, 5]
PASS: materialized/online references; causal positions; rescaling; partitions; shift stability; masked/invalid diagnostics
ATTENTION WALKTHROUGH PASS: CPU binary64 mathematics; no CUTLASS compilation or GPU execution
```

<!-- attention-walkthrough:capture:end -->

## 13 — Exercises with worked answers

??? question "Exercise 1: move one V row"

    Keep Q/K unchanged, but replace key two's V row by `[0,0]`. What are the noncausal outputs?

    **Answer:** The probabilities stay unchanged. Row zero becomes `[2/5,2/5]`; row one becomes `[1/5,4/5]`. Masking key two instead would change the denominator and is a different operation.

??? question "Exercise 2: identify a bad mask"

    At causal query zero, an implementation replaces excluded scores by zero. What does it compute?

    **Answer:** Its exponentials are `[2,1,1]`, so probabilities are `[1/2,1/4,1/4]`. Output becomes `[1.25,1.5]`, rather than `[1,0]`. Excluded entries must contribute no softmax mass.

??? question "Exercise 3: why save two row statistics?"

    Can a streaming implementation retain only l and omit m?

    **Answer:** Not with this recurrence and arbitrary logits. l is defined relative to m. A new maximum changes that reference and requires alpha to rescale old l and z. Another algorithm could store other equivalent information, but it still needs a well-defined stable representation.

??? question "Exercise 4: count full-matrix storage"

    A separate example has Nq=Nk=1024 and materializes both S and P in FP32. How many bytes do just those two matrices occupy per head?

    **Answer:** `2*1024*1024*4=8388608` bytes, or 8 MiB. This excludes Q/K/V/O, temporary storage, extra heads/batches and allocator overhead. Avoiding that materialization is not a claim that all other memory traffic disappears.

## Summary

- Define the Q/K and P/V reduction axes, positions, mask and conversion policy before choosing an FMHA kernel.
- Online attention carries a row maximum, scalar denominator and vector numerator. A new maximum rescales both accumulated quantities; mask before contributing to any statistic.
- Layouts map feature coordinates to storage independently of score coordinates. A logical transpose, valid address map or tiny correct equation does not establish a supported Hopper launch.
- The CPU oracle checks the mathematics. Continue to the pinned implementation and its separately stated build/validation scope in [Chapter 31](31-cutlass-hopper-fmha.md).
