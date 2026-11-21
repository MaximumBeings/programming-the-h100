# Complete attention source presentation — November 18, 2026

> Historical audit, scoped to its stated date and source revision. Later source corrections and outstanding GPU reruns are tracked in [the November 21 review repair record](../docs/review-fixes-2026-11-21.md). This note does not upgrade an old execution to a result for corrected source.

Chapter 31 previously linked to the pinned implementation without presenting its complete compilation unit. It now displays all 1,192 lines of `88_hopper_fmha.cu`, provides an individual download and a complete local-header project ZIP, and links the kernel, QK/PV mainloop, softmax and epilogue headers directly. A jump link near the top makes the listing discoverable.

All 24 original example files, including the runner, are copied byte-for-byte from CUTLASS commit `1afc6d355b8cb992398f17beb4425702f1d57ddd`; the root BSD license is included. The upstream README is stored as `README.upstream.txt` in the website assets to avoid treating upstream relative documentation links as book pages. Its original name is recorded in the manifest. A manifest identifies every file. The runner SHA-256 remains `b29f795d77867fbac7954c55c39042d7a5b145659f84742c568c305cfc47b4c5`.

Build instructions use the official pinned CMake target and explicitly copy the published project into a scratch checkout. They give supported D=64 forward and backward numerical-check commands. The chapter calls out the upstream banner's unsupported all-2048 example instead of recommending it.

`tools/verify_attention_source.py` checks file identities, equality of the visible full source and download apart from presentation trailing whitespace, and every ZIP member; deployment CI runs it. Every quoted example include was checked against the supplied project or pinned CUTLASS include directories. Strict build, internal links and displayed-source token preservation were checked. Formatting affects presentation only; downloaded source retains upstream bytes.

No GPU compilation or execution was performed. Hopper validation is deferred as previously agreed; the chapter does not invent GPU output or execution dates.
