# Final audit — two H100 GPUs

This fresh audit supplements the [original November 17 evidence](validation/2026-11-17/results.json). Historical records and chapter transcripts are preserved.

All **27 complete CUDA programs compiled**. Workload statuses: **26 PASS**, **1 hardware-blocked**. A prerequisite exit is not a successful workload.

[System and tool information](validation/2026-11-17-final-audit/system-info.log) · [Raw execution manifest](validation/2026-11-17-final-audit/results.json) · [Integrity inventory](validation/2026-11-17-final-audit/integrity.json). The system log records CUDA-visible discovery, GPU names, memory, driver, compiler, OS and topology.

Each command below is normalized to a local source filename and `example` executable. The raw manifest preserves the actual host paths, commands, stage exits and UTC timestamps. Run commands belong in the matching extracted source directory.

## Execution system

```text
UTC: 2026-11-17T23:33:18.920847+00:00
OS: Linux-6.8.0-1064-nvidia-x86_64-with-glibc2.39
Python: 3.12.3
PRETTY_NAME="Ubuntu 24.04.5 LTS"
Cuda compilation tools, release 12.8, V12.8.93
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
NVIDIA H100 80GB HBM3, 580.105.08, 81559 MiB
NVIDIA H100 80GB HBM3, 580.105.08, 81559 MiB
```

## Complete program results

### ex01_cluster_inspect.cu

**PASS** · UTC: `2026-11-17T23:33:20.254156+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:20.254398+00:00` · Run UTC: `2026-11-17T23:33:23.894201+00:00` · Finished UTC: `2026-11-17T23:33:24.565983+00:00`.

Source SHA-256: `ff9c3f47e5bc98d502ca02011672ee2fe024f52e93614d3eb91284953ce6e2f3`.

[Source](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex01_cluster_inspect/ex01_cluster_inspect.cu) · [Compiler output](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex01_cluster_inspect/compile.log) · [Runtime output](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex01_cluster_inspect/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_cluster_inspect.cu -o example
./example
```

### ex02_ptx_basics.cu

**PASS** · UTC: `2026-11-17T23:33:24.566141+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:24.566360+00:00` · Run UTC: `2026-11-17T23:33:25.710906+00:00` · Finished UTC: `2026-11-17T23:33:26.376286+00:00`.

Source SHA-256: `50b26d6098753bb1cf202e4482308bf86020d3e0eee4fad0b59f2cfcc2eef70f`.

[Source](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex02_ptx_basics/ex02_ptx_basics.cu) · [Compiler output](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex02_ptx_basics/compile.log) · [Runtime output](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex02_ptx_basics/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex02_ptx_basics.cu -o example
./example
```

### ex03_state_spaces.cu

**PASS** · UTC: `2026-11-17T23:33:26.376633+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:26.376875+00:00` · Run UTC: `2026-11-17T23:33:27.422786+00:00` · Finished UTC: `2026-11-17T23:33:27.996637+00:00`.

Source SHA-256: `1ae7bdef0dbe25d98cc52b89f78a3b0562def59c985a9a8db1286c79bee47fc8`.

[Source](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex03_state_spaces/ex03_state_spaces.cu) · [Compiler output](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex03_state_spaces/compile.log) · [Runtime output](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex03_state_spaces/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex03_state_spaces.cu -o example
./example
```

### ex04_ptx_pointers.cu

**PASS** · UTC: `2026-11-17T23:33:27.996910+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:27.997073+00:00` · Run UTC: `2026-11-17T23:33:29.313664+00:00` · Finished UTC: `2026-11-17T23:33:29.874379+00:00`.

Source SHA-256: `77953a91b667b873d64296e0437bb2ca69234210f9dbdb9998125ed3a7512c8c`.

[Source](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex04_ptx_pointers/ex04_ptx_pointers.cu) · [Compiler output](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex04_ptx_pointers/compile.log) · [Runtime output](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex04_ptx_pointers/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex04_ptx_pointers.cu -o example
./example
```

### ex05_ptx_datatypes.cu

**PASS** · UTC: `2026-11-17T23:33:29.874617+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:29.874824+00:00` · Run UTC: `2026-11-17T23:33:31.474149+00:00` · Finished UTC: `2026-11-17T23:33:32.039296+00:00`.

Source SHA-256: `711a86340850bdbae0dfd8b9d809ad27e7c6da2b057253d8f667a698e9630220`.

[Source](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex05_ptx_datatypes/ex05_ptx_datatypes.cu) · [Compiler output](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex05_ptx_datatypes/compile.log) · [Runtime output](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex05_ptx_datatypes/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex05_ptx_datatypes.cu -o example
./example
```

### ex06_comprehensive.cu

**PASS** · UTC: `2026-11-17T23:33:32.039572+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:32.039804+00:00` · Run UTC: `2026-11-17T23:33:35.426680+00:00` · Finished UTC: `2026-11-17T23:33:36.083902+00:00`.

Source SHA-256: `4c5fb40a537b3c5dfc2190023302f953f98867349e486a5daeba6e73e823be8d`.

[Source](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex06_comprehensive/ex06_comprehensive.cu) · [Compiler output](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex06_comprehensive/compile.log) · [Runtime output](validation/2026-11-17-final-audit/02-clusters-inline-ptx/ex06_comprehensive/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex06_comprehensive.cu -o example
./example
```

### ex01_mbarrier_basic.cu

**PASS** · UTC: `2026-11-17T23:33:36.084170+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:36.084382+00:00` · Run UTC: `2026-11-17T23:33:37.123884+00:00` · Finished UTC: `2026-11-17T23:33:37.686317+00:00`.

Source SHA-256: `075961dac147dd0c2ef4a5a288af51c6cc6f0330364bb8fbf4b62cbf4c5aef92`.

[Source](validation/2026-11-17-final-audit/03-asynchronicity-mbarrier/ex01_mbarrier_basic/ex01_mbarrier_basic.cu) · [Compiler output](validation/2026-11-17-final-audit/03-asynchronicity-mbarrier/ex01_mbarrier_basic/compile.log) · [Runtime output](validation/2026-11-17-final-audit/03-asynchronicity-mbarrier/ex01_mbarrier_basic/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_mbarrier_basic.cu -o example
./example
```

### ex02_war_prevention.cu

**PASS** · UTC: `2026-11-17T23:33:37.686605+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:37.686807+00:00` · Run UTC: `2026-11-17T23:33:38.711863+00:00` · Finished UTC: `2026-11-17T23:33:39.490572+00:00`.

Source SHA-256: `852967b65c0f83348cffac11df0ed9fbbed40802aa730944f31564efda5eb2d0`.

[Source](validation/2026-11-17-final-audit/03-asynchronicity-mbarrier/ex02_war_prevention/ex02_war_prevention.cu) · [Compiler output](validation/2026-11-17-final-audit/03-asynchronicity-mbarrier/ex02_war_prevention/compile.log) · [Runtime output](validation/2026-11-17-final-audit/03-asynchronicity-mbarrier/ex02_war_prevention/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex02_war_prevention.cu -o example
./example
```

### ex01_tma_basic.cu

**PASS** · UTC: `2026-11-17T23:33:39.490868+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:39.491084+00:00` · Run UTC: `2026-11-17T23:33:40.566260+00:00` · Finished UTC: `2026-11-17T23:33:41.213013+00:00`.

Source SHA-256: `c94e84ae2a4cf9dbd8d2c0c8a65c40399613eca6a5a40b262109c9a2f5f57caa`.

[Source](validation/2026-11-17-final-audit/04-cutensormap-tma-swizzling/ex01_tma_basic/ex01_tma_basic.cu) · [Compiler output](validation/2026-11-17-final-audit/04-cutensormap-tma-swizzling/ex01_tma_basic/compile.log) · [Runtime output](validation/2026-11-17-final-audit/04-cutensormap-tma-swizzling/ex01_tma_basic/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_tma_basic.cu -o example -lcuda
./example
```

### ex02_tma_3d.cu

**PASS** · UTC: `2026-11-17T23:33:41.213313+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:41.213487+00:00` · Run UTC: `2026-11-17T23:33:42.549323+00:00` · Finished UTC: `2026-11-17T23:33:43.106113+00:00`.

Source SHA-256: `b80719ce45119fa91323bc610a68e7aeb84945ccfa8864fc3f2ee06cc594cab3`.

[Source](validation/2026-11-17-final-audit/04-cutensormap-tma-swizzling/ex02_tma_3d/ex02_tma_3d.cu) · [Compiler output](validation/2026-11-17-final-audit/04-cutensormap-tma-swizzling/ex02_tma_3d/compile.log) · [Runtime output](validation/2026-11-17-final-audit/04-cutensormap-tma-swizzling/ex02_tma_3d/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex02_tma_3d.cu -o example -lcuda
./example
```

### ex01_bulk_raw.cu

**PASS** · UTC: `2026-11-17T23:33:43.106426+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:43.106625+00:00` · Run UTC: `2026-11-17T23:33:44.119713+00:00` · Finished UTC: `2026-11-17T23:33:44.680974+00:00`.

Source SHA-256: `d3205e1cc28e486f95c1923b3d347bc08bb8f9e52ae4324a8171bef168f22fbd`.

[Source](validation/2026-11-17-final-audit/05-cp-async-bulk-l2-hinting/ex01_bulk_raw/ex01_bulk_raw.cu) · [Compiler output](validation/2026-11-17-final-audit/05-cp-async-bulk-l2-hinting/ex01_bulk_raw/compile.log) · [Runtime output](validation/2026-11-17-final-audit/05-cp-async-bulk-l2-hinting/ex01_bulk_raw/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_bulk_raw.cu -o example
./example
```

### ex02_reduce_async.cu

**PASS** · UTC: `2026-11-17T23:33:44.681325+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:44.681474+00:00` · Run UTC: `2026-11-17T23:33:45.681496+00:00` · Finished UTC: `2026-11-17T23:33:46.345168+00:00`.

Source SHA-256: `52617e1a22a4619a1fd65dc92d4eac529b3bd02ee2ea34803ab51213f829f39a`.

[Source](validation/2026-11-17-final-audit/05-cp-async-bulk-l2-hinting/ex02_reduce_async/ex02_reduce_async.cu) · [Compiler output](validation/2026-11-17-final-audit/05-cp-async-bulk-l2-hinting/ex02_reduce_async/compile.log) · [Runtime output](validation/2026-11-17-final-audit/05-cp-async-bulk-l2-hinting/ex02_reduce_async/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex02_reduce_async.cu -o example
./example
```

### ex01_wgmma_basic.cu

**PASS** · UTC: `2026-11-17T23:33:46.345505+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:46.345755+00:00` · Run UTC: `2026-11-17T23:33:48.039121+00:00` · Finished UTC: `2026-11-17T23:33:48.695511+00:00`.

Source SHA-256: `282064c18eedf80395d5c5ed41bc0d1e4160c8d3886bf831c90b68628b84ad04`.

[Source](validation/2026-11-17-final-audit/06-wgmma-part-1/ex01_wgmma_basic/ex01_wgmma_basic.cu) · [Compiler output](validation/2026-11-17-final-audit/06-wgmma-part-1/ex01_wgmma_basic/compile.log) · [Runtime output](validation/2026-11-17-final-audit/06-wgmma-part-1/ex01_wgmma_basic/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_wgmma_basic.cu -o example
./example
```

### ex01_wgmma_pipeline.cu

**PASS** · UTC: `2026-11-17T23:33:48.695895+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:48.696129+00:00` · Run UTC: `2026-11-17T23:33:50.364496+00:00` · Finished UTC: `2026-11-17T23:33:51.032181+00:00`.

Source SHA-256: `76960b5ead233cd05431fe9c4c7847f859e3270a7407461f4dbe91006c1c2790`.

[Source](validation/2026-11-17-final-audit/07-wgmma-part-2/ex01_wgmma_pipeline/ex01_wgmma_pipeline.cu) · [Compiler output](validation/2026-11-17-final-audit/07-wgmma-part-2/ex01_wgmma_pipeline/compile.log) · [Runtime output](validation/2026-11-17-final-audit/07-wgmma-part-2/ex01_wgmma_pipeline/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_wgmma_pipeline.cu -o example
./example
```

### ex01_roofline.cu

**PASS** · UTC: `2026-11-17T23:33:51.032534+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:51.032798+00:00` · Run UTC: `2026-11-17T23:33:52.077059+00:00` · Finished UTC: `2026-11-17T23:33:52.641842+00:00`.

Source SHA-256: `d4e66adac175c64e13c8761c308a1a5c2151164a5505cd6959d04b8abab61449`.

[Source](validation/2026-11-17-final-audit/08-kernel-design-part-1/ex01_roofline/ex01_roofline.cu) · [Compiler output](validation/2026-11-17-final-audit/08-kernel-design-part-1/ex01_roofline/compile.log) · [Runtime output](validation/2026-11-17-final-audit/08-kernel-design-part-1/ex01_roofline/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex01_roofline.cu -o example
./example
```

### ex02_warp_spec.cu

**PASS** · UTC: `2026-11-17T23:33:52.642253+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:52.642439+00:00` · Run UTC: `2026-11-17T23:33:53.647666+00:00` · Finished UTC: `2026-11-17T23:33:54.296581+00:00`.

Source SHA-256: `fb1010dd4b8597fc35c1f99944db4a44e052228546c7f1e794f78560043be675`.

[Source](validation/2026-11-17-final-audit/08-kernel-design-part-1/ex02_warp_spec/ex02_warp_spec.cu) · [Compiler output](validation/2026-11-17-final-audit/08-kernel-design-part-1/ex02_warp_spec/compile.log) · [Runtime output](validation/2026-11-17-final-audit/08-kernel-design-part-1/ex02_warp_spec/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex02_warp_spec.cu -o example
./example
```

### ex03_setmaxnreg.cu

**PASS** · UTC: `2026-11-17T23:33:54.297038+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:54.297214+00:00` · Run UTC: `2026-11-17T23:33:55.301591+00:00` · Finished UTC: `2026-11-17T23:33:55.863438+00:00`.

Source SHA-256: `c640beecccfe53b9968d7e5f14e939dfaa0966c45829c480694bfa45556e2292`.

[Source](validation/2026-11-17-final-audit/08-kernel-design-part-1/ex03_setmaxnreg/ex03_setmaxnreg.cu) · [Compiler output](validation/2026-11-17-final-audit/08-kernel-design-part-1/ex03_setmaxnreg/compile.log) · [Runtime output](validation/2026-11-17-final-audit/08-kernel-design-part-1/ex03_setmaxnreg/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex03_setmaxnreg.cu -o example
./example
```

### ex04_circular_buffer.cu

**PASS** · UTC: `2026-11-17T23:33:55.863890+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:55.864082+00:00` · Run UTC: `2026-11-17T23:33:57.073830+00:00` · Finished UTC: `2026-11-17T23:33:57.638973+00:00`.

Source SHA-256: `3ea5fd796a9fa77c9d2c002bdf8a440c9071f248946514164134ea66c41c4665`.

[Source](validation/2026-11-17-final-audit/08-kernel-design-part-1/ex04_circular_buffer/ex04_circular_buffer.cu) · [Compiler output](validation/2026-11-17-final-audit/08-kernel-design-part-1/ex04_circular_buffer/compile.log) · [Runtime output](validation/2026-11-17-final-audit/08-kernel-design-part-1/ex04_circular_buffer/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex04_circular_buffer.cu -o example
./example
```

### ex05_mbarrier.cu

**PASS** · UTC: `2026-11-17T23:33:57.639389+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:57.639630+00:00` · Run UTC: `2026-11-17T23:33:58.684751+00:00` · Finished UTC: `2026-11-17T23:33:59.347429+00:00`.

Source SHA-256: `c6a18427946c38f4305dc305ece5ade0461dffe09bce0fae6c87f3d53eefe046`.

[Source](validation/2026-11-17-final-audit/09-kernel-design-part-2/ex05_mbarrier/ex05_mbarrier.cu) · [Compiler output](validation/2026-11-17-final-audit/09-kernel-design-part-2/ex05_mbarrier/compile.log) · [Runtime output](validation/2026-11-17-final-audit/09-kernel-design-part-2/ex05_mbarrier/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex05_mbarrier.cu -o example
./example
```

### ex06_pingpong.cu

**PASS** · UTC: `2026-11-17T23:33:59.347884+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:33:59.348062+00:00` · Run UTC: `2026-11-17T23:34:00.440175+00:00` · Finished UTC: `2026-11-17T23:34:00.651516+00:00`.

Source SHA-256: `d1b6baf573465e1a90bef7d3a51c91508fbe818dbb311a49c8fe532beb3930cd`.

[Source](validation/2026-11-17-final-audit/09-kernel-design-part-2/ex06_pingpong/ex06_pingpong.cu) · [Compiler output](validation/2026-11-17-final-audit/09-kernel-design-part-2/ex06_pingpong/compile.log) · [Runtime output](validation/2026-11-17-final-audit/09-kernel-design-part-2/ex06_pingpong/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex06_pingpong.cu -o example
./example
```

### ex07_epilogue.cu

**PASS** · UTC: `2026-11-17T23:34:00.651993+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:34:00.652181+00:00` · Run UTC: `2026-11-17T23:34:01.725195+00:00` · Finished UTC: `2026-11-17T23:34:02.282790+00:00`.

Source SHA-256: `bfdc7b7c4d6b42fb42aa5ee46b9ca7611ca2b0ba6be64c12bbdb05a7baff5bff`.

[Source](validation/2026-11-17-final-audit/09-kernel-design-part-2/ex07_epilogue/ex07_epilogue.cu) · [Compiler output](validation/2026-11-17-final-audit/09-kernel-design-part-2/ex07_epilogue/compile.log) · [Runtime output](validation/2026-11-17-final-audit/09-kernel-design-part-2/ex07_epilogue/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex07_epilogue.cu -o example
./example
```

### ex08_comprehensive.cu

**PASS** · UTC: `2026-11-17T23:34:02.283337+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:34:02.283546+00:00` · Run UTC: `2026-11-17T23:34:03.937963+00:00` · Finished UTC: `2026-11-17T23:34:04.636912+00:00`.

Source SHA-256: `7c9011cfcbcf2cc91ff61610b8b55e4d3548700e8435178fa9967dff80fd0964`.

[Source](validation/2026-11-17-final-audit/09-kernel-design-part-2/ex08_comprehensive/ex08_comprehensive.cu) · [Compiler output](validation/2026-11-17-final-audit/09-kernel-design-part-2/ex08_comprehensive/compile.log) · [Runtime output](validation/2026-11-17-final-audit/09-kernel-design-part-2/ex08_comprehensive/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex08_comprehensive.cu -o example
./example
```

### ex09_streamk_roles.cu

**PASS** · UTC: `2026-11-17T23:34:04.637356+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:34:04.637583+00:00` · Run UTC: `2026-11-17T23:34:08.031768+00:00` · Finished UTC: `2026-11-17T23:34:08.702461+00:00`.

Source SHA-256: `fd8d2d61ed5f5245d114196d678918a58f33401445342963fdaaafef32499fcf`.

[Source](validation/2026-11-17-final-audit/10-stream-k-scheduling/ex09_streamk_roles/ex09_streamk_roles.cu) · [Compiler output](validation/2026-11-17-final-audit/10-stream-k-scheduling/ex09_streamk_roles/compile.log) · [Runtime output](validation/2026-11-17-final-audit/10-stream-k-scheduling/ex09_streamk_roles/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex09_streamk_roles.cu -o example
./example
```

### ex10_dependent_launch.cu

**PASS** · UTC: `2026-11-17T23:34:08.702960+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:34:08.703110+00:00` · Run UTC: `2026-11-17T23:34:09.717502+00:00` · Finished UTC: `2026-11-17T23:34:10.381720+00:00`.

Source SHA-256: `36b6313722fe43c219e825b2fe4c4e0b5cbb1630ceb02d3fe1624c2c242a7640`.

[Source](validation/2026-11-17-final-audit/10-stream-k-scheduling/ex10_dependent_launch/ex10_dependent_launch.cu) · [Compiler output](validation/2026-11-17-final-audit/10-stream-k-scheduling/ex10_dependent_launch/compile.log) · [Runtime output](validation/2026-11-17-final-audit/10-stream-k-scheduling/ex10_dependent_launch/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex10_dependent_launch.cu -o example
./example
```

### ex11_nvlink_bandwidth.cu

**PASS** · UTC: `2026-11-17T23:36:22.842493+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:36:22.842759+00:00` · Run UTC: `2026-11-17T23:36:24.075201+00:00` · Finished UTC: `2026-11-17T23:36:26.908842+00:00`.

Source SHA-256: `67e71f448fd51c1e130ef1c87abeece5d694e5546f6c32c837ca44db30b6c305`.

[Source](validation/2026-11-17-final-audit/11-multi-gpu-nccl/ex11_nvlink_bandwidth/ex11_nvlink_bandwidth.cu) · [Compiler output](validation/2026-11-17-final-audit/11-multi-gpu-nccl/ex11_nvlink_bandwidth/compile.log) · [Runtime output](validation/2026-11-17-final-audit/11-multi-gpu-nccl/ex11_nvlink_bandwidth/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex11_nvlink_bandwidth.cu -o example
./example
```

### ex12_nccl_allreduce.cu

**PASS** · UTC: `2026-11-17T23:36:26.909012+00:00` · compile exit: `0` · run exit: `0`.

Compile UTC: `2026-11-17T23:36:26.909178+00:00` · Run UTC: `2026-11-17T23:36:28.640907+00:00` · Finished UTC: `2026-11-17T23:36:30.353034+00:00`.

Source SHA-256: `f0cc87820667319cc733016feb646d72e31d4bf2dd26a20b8329545b7c062889`.

[Source](validation/2026-11-17-final-audit/11-multi-gpu-nccl/ex12_nccl_allreduce/ex12_nccl_allreduce.cu) · [Compiler output](validation/2026-11-17-final-audit/11-multi-gpu-nccl/ex12_nccl_allreduce/compile.log) · [Runtime output](validation/2026-11-17-final-audit/11-multi-gpu-nccl/ex12_nccl_allreduce/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex12_nccl_allreduce.cu -o example -lnccl
./example
```

### ex13_hybrid_parallel_training.cu

**BLOCKED_HARDWARE** · UTC: `2026-11-17T23:34:15.805469+00:00` · compile exit: `0` · run exit: `unexecuted`.

Compile UTC: `2026-11-17T23:34:15.805712+00:00` · Run UTC: `not executed` · Finished UTC: `2026-11-17T23:34:17.837357+00:00`.

Source SHA-256: `99f1b0a3238b2833c49ed65171cba66cd446991a1be3e807a1645a7f408de9f3`.

[Source](validation/2026-11-17-final-audit/11-multi-gpu-nccl/ex13_hybrid_parallel_training/ex13_hybrid_parallel_training.cu) · [Compiler output](validation/2026-11-17-final-audit/11-multi-gpu-nccl/ex13_hybrid_parallel_training/compile.log) · [Runtime output](validation/2026-11-17-final-audit/11-multi-gpu-nccl/ex13_hybrid_parallel_training/run.log).

```bash
nvcc -std=c++17 -arch=sm_90a -O2 ex13_hybrid_parallel_training.cu -o example -lnccl -lcublas -Xcompiler -pthread
./example
```

## Nonuniform epilogue audit

Status: **PASSED** · UTC: `2026-11-17T23:33:36.701318+00:00`. This opt-in variant uses 4,096 unique exactly representable expected outputs and preserves the device implementation. It strengthens layout coverage; it is not a new performance benchmark.

[Manifest](validation/2026-11-17-final-audit/nonuniform/manifest.json) · [Variant source](validation/2026-11-17-final-audit/nonuniform/ch07_nonuniform_epilogue.cu) · [Compiler output](validation/2026-11-17-final-audit/nonuniform/compile.log) · [Runtime output](validation/2026-11-17-final-audit/nonuniform/run.log).

Compute Sanitizer memcheck of this nonuniform variant reports **ERROR SUMMARY: 0 errors**. [Captured memcheck log](validation/2026-11-17-final-audit/nonuniform/memcheck.log). This is memory-access checking for that execution, not proof of every asynchronous protocol or a performance measurement.

## Online-softmax numerical reference

All **12 cases passed** against PyTorch math SDPA, including causal masks and partial final tiles. Execution metadata is retained verbatim in the captured log. [Captured output](validation/2026-11-17-final-audit/attention/run.log) · [Executed reference source](validation/2026-11-17-final-audit/attention/validate_attention.py). This verifies the recurrence for those cases, not the FA3 skeleton’s synchronization or published throughput.

## Verification scope

CI verifies source and evidence hashes, workload status consistency, captured-log presence and this summary. CI does not rerun CUDA or establish untested numerical coverage. The remaining eight-GPU workload requires the corresponding hardware.
