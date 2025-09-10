Title: Ring Buffer KV Cache – Compile‑Safe Design and Staged Rollout Plan

Overview
- This proposal defines a safe, testable path to introduce ring buffer semantics for the KV cache while avoiding the prior black‑frame regressions and compile retraces.
- It is written in the same style as `mxfp_inference_transition.md` and `fp8_kv_flexattention.md`, with explicit stages, validation gates, commands, and rollback.

Background and prior incidents
- Earlier attempts to combine a ring buffer with tail‑only reads produced black frames immediately after the first frame.
- Root causes suspected in prior logs:
  - Scale/index misalignment for per‑token (timewise) scales vs K/V ring positions.
  - Dynamic indexing inside `torch.compile` inducing retraces and unstable runtime.
- A revert to the legacy roll‑based window restored visuals; performance and stability were better with the roll path.

Goals
- Maintain visual correctness while enabling ring semantics for the KV cache.
- Keep attention compute and activations in BF16; only storage/layout changes in the cache.
- Ensure compile‑friendly behavior (no frequent retraces) during decode.

Non‑negotiable invariants
- Positional correctness: `get_offset(layer)` must monotonically increase by tokens written; logical token order must match RoPE expectations.
- Layout and contiguity: K/V presented to FlexAttention must be `[B,H,T,D]` contiguous BF16.
- Scale alignment: for FP8‑like storage with per‑token scales `[B,H,T]`, the scale slice must align exactly with the corresponding time slices of K/V.
- Constant op shapes in compiled paths: avoid data‑dependent control flow or dynamic tensor shapes inside compiled forward.

Design options
1) Roll‑on‑update (compile‑safe, default)
   - Maintain a fixed contiguous slab `[B,H,T,D]` and on each update perform a constant‑shift `torch.roll(..., shifts=-tokens_per_frame, dims=2)` on K/V and scales, then write at the tail.
   - Pros: constant op shapes and shift magnitude; plays well with compile; proven to be stable.
   - Cons: performs a full‑window copy per step; acceptable for our sizes and current stability goals.

2) Pointer ring with two‑copy “gather to scratch” (experimental)
   - Maintain a logical write pointer; on read, assemble a contiguous window into a preallocated BF16 scratch by copying two fixed slices (tail then head). Always perform two copies to keep the compiled graph stable.
   - Pros: avoids full‑window roll per update; amortizes cost to two linear copies per read.
   - Cons: introduces data‑dependent indices; must be implemented to be compile‑friendly (see Stage C). Easy to misalign scales and induce black frames.

3) Fused ring kernels (future)
   - Implement a fused kernel to assemble contiguous views from ring storage with scales applied during dequant.
   - Pros: minimal overhead; avoids Python control flow entirely.
   - Cons: non‑trivial engineering; out of scope for the initial rollout.

Staged rollout (testable phases)
Stage A – Baseline parity (roll‑on‑update only)
- Keep the existing roll‑on‑update design for K/V and per‑token scales (timewise) in `QuantizedStaticCache` and/or `StaticCache`.
- Unit tests:
  - KV equivalence: feed synthetic sequences; after N updates, assert `get()` windows from roll‑cache match ground truth concatenation.
  - Offset correctness: assert `get_offset()` increases by `tokens_per_frame` per step, equals logical window length in tokens.
  - Scale alignment: assert that the per‑token scales slice indices match K/V time indices on every update.
- Integration checks:
  - Short decode rollouts with compile ON; verify visuals and no black frames.
  - Profile to ensure no additional retraces.
- Gate A (go/no‑go): proceed only if visuals match baseline and no compile regressions are observed.

Stage B – BF16 roll cache parity in FP8 path
- For layers using FP8 storage, ensure timewise per‑token scales `[B,H,T]` and the i8 slabs roll together with identical shifts and tail writes.
- Add instrumentation:
  - Log write pointer (logical), effective filled tokens, and slices on update/get under `OWL_PROFILE_KV=1`.
  - Track NaN/Inf scale fraction (should be 0) and saturation counts (|q|=127) per layer to catch mis‑scaling early.
- Unit tests:
  - Round‑trip small tensors through quant→ring‑roll→dequant and compare to BF16 baseline; assert MAE within tolerance.
- Gate B: proceed if unit and integration tests pass and visuals remain stable.

Stage C – Pointer ring (experimental, guarded, compile‑safe)
- Introduce a logical write pointer for K/V and scales but keep a compile‑friendly read:
  - Always perform two fixed‑sequence memcopies into a preallocated BF16 scratch window regardless of wrap (copy lengths vary but ops count stays constant).
  - Keep these copies inside the compiled graph only if they do not trigger retraces; otherwise, keep pointer assembly outside the compiled forward (not preferred), or disable compile for the cache path only.
- Safeguards:
  - Assert pointer modulo T matches expected logical offset (`offset % T`).
  - Assert that scale slices are built from the exact same indices as K/V slices.
  - Provide a checksum mode: for N small steps, compare the assembled BF16 K/V vs a reference roll cache.
- Gate C: only promote beyond local experiments if there are zero visual regressions, no retraces, and measurable reduction in memory movement vs roll.

Stage D – Optional tail‑only reads (revisit)
- If masks guarantee local windows, dequantize/assemble only the tail tokens used by FlexAttention.
- This was implicated in earlier black frames; only attempt after Stages A–C are stable, with strong assertions and unit tests on index math.
- Gate D: proceed only if pointer ring is fully stable and mask parity is validated for toy cases.

Implementation notes and file‑by‑file plan
- `owl_wms/nn/kv_cache.py`
  - Stage A/B: keep roll‑on‑update semantics with constant shift `-tokens_per_frame` for K/V and timewise scales.
  - Stage C: add an optional pointer‑ring mode behind a flag; implement `get()` that assembles a contiguous BF16 view using two fixed copies into a preallocated scratch buffer (per layer). Do not change `Attn.forward` signature.
  - Add strict assertions in debug: pointer vs offset, scale/KV slice parity, `[B,H,T,D]` contiguity.

- `owl_wms/quant/fp8_kv.py`
  - Keep per‑token (timewise) quant/dequant functions and ensure they operate in BF16 output.
  - Add optional counters for saturation and scale floors to aid debugging.

- `owl_wms/nn/attn.py`
  - No functional changes; continue to request BF16 `[B,H,T,D]` from the cache.
  - If needed for Stage D, introduce an optional `get_slice(layer, start, end)` API only after stability is proven.

- `inference/causvid_pipeline.py`
  - Wire flags; print whether pointer‑ring is active. Keep compile enabled; ensure cache methods are compile‑friendly.
  - Maintain existing profiling prints; extend with pointer/logical window diagnostics when `OWL_PROFILE_KV=1`.

Flags and environment variables
- `OWL_RING_MODE=roll|pointer` (default `roll`)
- `OWL_RING_GUARD=1` enable assertions and index/scale parity checks
- `OWL_RING_SCRATCH=1` use preallocated BF16 scratch buffers for `get()` assembly (Stage C)
- `OWL_KV_TAIL_ONLY=0/1` tail read optimization (keep 0 until Stage D)
- Existing KV flags remain in effect: `OWL_FP8_KV`, `OWL_K_FP8`, `OWL_KV_LATE_LAYERS`, `OWL_KV_STORAGE`, `OWL_KV_BITS`, `OWL_PROFILE_KV`, `OWL_PROFILE_KV_FIRST`, `OWL_PROFILE_KV_EVERY`.

Validation plan
1) Unit tests (toy shapes)
   - Roll vs ground truth concat equivalence across N steps.
   - Pointer ring assembly equivalence vs roll for N steps.
   - Per‑token scale alignment checks; saturation/NaN counters == 0.

2) Integration
   - Two short decode clips (stand‑still, motion) with compile ON.
   - Collect `[KV-Profile]` lines: `attn_ms`, `decoder_ms`, `q_ms`, `dq_ms`, `max_reserved`.
   - Ensure no black frames; no unusual lag spikes (retrace signatures).

3) A/B gates
   - Gate A: Stage A parity (roll) must be visually identical to current stable path; no retraces.
   - Gate B: FP8 timewise roll must preserve visuals; `q_ms/dq_ms` stable.
   - Gate C: Pointer ring must show no retraces and stable visuals in 3+ scenes before enabling by default.
   - Gate D: Tail‑only reads only after C; must match mask semantics exactly on toy tests.

Rollback strategy
- Immediate disable: `OWL_RING_MODE=roll` (or `OWL_FP8_KV=0` to exit FP8 path entirely).
- Remove tail‑only reads: `OWL_KV_TAIL_ONLY=0`.
- Keep K in BF16: `OWL_K_FP8=0`.

Commands (examples)
Baseline BF16 KV
```bash
OWL_PROFILE_KV=1 OWL_FP8_KV=0 python -m inference.game_cv
```

Roll ring (compile‑safe default), FP8 V‑only
```bash
OWL_COMPILE=1 OWL_PROFILE_KV=1 \
OWL_FP8_KV=1 OWL_K_FP8=0 OWL_RING_MODE=roll \
python -m inference.game_cv
```

Pointer ring (experimental), FP8 V‑only
```bash
OWL_COMPILE=1 OWL_PROFILE_KV=1 \
OWL_FP8_KV=1 OWL_K_FP8=0 OWL_RING_MODE=pointer OWL_RING_GUARD=1 OWL_RING_SCRATCH=1 \
python -m inference.game_cv
```

Tail‑only reads (do not enable until Stage D)
```bash
OWL_COMPILE=1 OWL_PROFILE_KV=1 \
OWL_FP8_KV=1 OWL_K_FP8=0 OWL_RING_MODE=pointer OWL_KV_TAIL_ONLY=1 \
python -m inference.game_cv
```

Instrumentation checklist
- On update: `layer`, `new_len`, `offset`, `shift_applied`, `write_range`, `filled_tokens`.
- On get: `layer`, `logical_len`, `wp` (if pointer mode), `assembled_ranges` (two slices), contiguity checks.
- Scales: NaN fraction, min/mean/max per layer every N frames; saturation counts.

Risks and mitigations
- Black frames from misalignment:
  - Mitigation: per‑token scales must roll with K/V together; add parity assertions; enable `OWL_RING_GUARD` in dev.
- Compile retraces from dynamic indexing:
  - Mitigation: default to roll‑on‑update with constant shift; in pointer mode, always execute the same two‑copy sequence into scratch to reduce graph variance.
- Memory movement from roll:
  - Mitigation: accept initially for stability; revisit pointer mode or fused kernels only after correctness is locked.

Success criteria
- Visual correctness matches baseline; no black frames across test scenes.
- No compile retraces in steady state.
- Optional pointer mode shows measurable reduction in K/V movement without regressions.

Next steps
- Implement Stage A/B assertions and instrumentation; land behind flags.
- Add unit tests for ring semantics and scale alignment.
- Trial Stage C locally with guards; promote only if stable.

Run log: Stage A smoke test (compile ON)
- Command:
```bash
OWL_COMPILE=1 OWL_PROFILE_KV=1 OWL_FP8_KV=1 OWL_K_FP8=0 OWL_RING_MODE=roll OWL_RING_GUARD=1 python -m inference.game_cv
```
- Observations:
  - Cold compile phase present; then steady‑state `attn1/attn2≈10–12 ms`, `decoder_ms≈19–20 ms`, pipeline FPS ≈ 42–44, total FPS ≈ 23–24.
  - `q_ms/dq_ms=0.00` (as expected with compile ON; quant/dequant fused/inlined).
  - `max_reserved` increased from ~3.3 GB to ~7.7 GB post‑prefill (allocator retention, consistent with prior notes).
- Raw snippet:
```text
[Kernels] sdp_kernel namespace present; SDPA backends configurable
[KV-Profile] prefill_s=12.748s, max_reserved=6770.0MB, fp8_kv=True
[KV-Profile] step1_s=7.994s (attn1=6288.57ms), step2_s=6.759s (attn2=7160.75ms), q_ms=0.00, dq_ms=0.00, max_reserved=6976.0MB
[KV-Profile] accum: attn_total_ms=0.00, mlp_total_ms=0.00
[KV-Profile] decoder_ms=1656.36ms
[19:43:34] FPS (total):   0.1 | FPS (pipeline):   0.1 | Latency pipeline: 13450.8 ms | Latency draw:    3.1 ms
[KV-Profile] step1_s=6.043s (attn1=6578.17ms), step2_s=6.732s (attn2=7171.60ms), q_ms=0.00, dq_ms=0.00, max_reserved=3198.0MB
[KV-Profile] decoder_ms=543.35ms
[19:43:48] FPS (total):   0.1 | FPS (pipeline):   0.1 | Latency pipeline: 13750.8 ms | Latency draw:    1.2 ms
[KV-Profile] step1_s=0.115s (attn1=123.91ms), step2_s=0.101s (attn2=108.08ms), q_ms=0.00, dq_ms=0.00, max_reserved=3292.0MB
[KV-Profile] decoder_ms=19.92ms
[19:43:49] FPS (total):  22.7 | FPS (pipeline):  41.3 | Latency pipeline:   23.6 ms | Latency draw:    1.4 ms
[KV-Profile] step1_s=0.011s (attn1=11.84ms), step2_s=0.012s (attn2=12.52ms), q_ms=0.00, dq_ms=0.00, max_reserved=3292.0MB
[KV-Profile] decoder_ms=19.88ms
[19:43:50] FPS (total):  23.5 | FPS (pipeline):  42.7 | Latency pipeline:   22.0 ms | Latency draw:    1.2 ms
[KV-Profile] step1_s=0.010s (attn1=10.52ms), step2_s=0.010s (attn2=10.93ms), q_ms=0.00, dq_ms=0.00, max_reserved=3292.0MB
[KV-Profile] decoder_ms=19.78ms
...
[KV-Profile] prefill_s=0.135s, max_reserved=7450.0MB, fp8_kv=True
...
[KV-Profile] step1_s=0.010s (attn1=10.77ms), step2_s=0.010s (attn2=10.93ms), q_ms=0.00, dq_ms=0.00, max_reserved=7656.0MB
[KV-Profile] decoder_ms=19.81ms
...
[KV-Profile] step1_s=0.010s (attn1=10.79ms), step2_s=0.010s (attn2=11.02ms), q_ms=0.00, dq_ms=0.00, max_reserved=7656.0MB
[KV-Profile] decoder_ms=19.86ms
```

Recommended next step (Stage A validation)
- Emit Stage A ring invariants by running a short compile‑OFF clip to print `update_done/get` guards we added:
```bash
OWL_COMPILE=0 OWL_PROFILE_KV=1 OWL_PROFILE_KV_FIRST=3 OWL_PROFILE_KV_EVERY=50 \
OWL_FP8_KV=1 OWL_K_FP8=0 OWL_RING_MODE=roll OWL_RING_GUARD=1 python -m inference.game_cv
```
- Expect lines like `update_done: layer=..., new_len=..., offset=..., filled=..., wp=...` and `get: layer=..., logical_len=..., filled=..., wp=...` with no assertion failures. If clean, Stage A is validated and we can proceed to Stage C experiments.

Run log: Stage A invariants (compile OFF)
- Command:
```bash
OWL_COMPILE=0 OWL_PROFILE_KV=1 OWL_PROFILE_KV_FIRST=3 OWL_PROFILE_KV_EVERY=50 \
OWL_FP8_KV=1 OWL_K_FP8=0 OWL_RING_MODE=roll OWL_RING_GUARD=1 python -m inference.game_cv
```
- Observations:
  - Printed `get/update_done` lines across all layers with stable `filled=3840`, `wp=0`, `logical_len` alternating between `3904` (get with new tokens logically appended) and `3840` (post‑update window), matching expectations for `[T=3840, new_len=64]` updates.
  - No assertion failures; offsets advance by `+64` each update (e.g., `4992 → 5056 → … → 6272`).
  - With compile OFF, `q_ms≈70ms`, `dq_ms≈92ms` were visible, consistent with prior notes that compile fuses/inlines Q/DQ.
- Raw snippet (excerpt):
```text
[KV-Profile] get: layer=0, logical_len=3904, filled=3840, wp=0
[KV-Profile] update_done: layer=0, new_len=64, offset=5056, filled=3840, wp=0
...
[KV-Profile] step1_s=0.040s (attn1=43.26ms), step2_s=0.031s (attn2=33.24ms), q_ms=70.28, dq_ms=92.60, max_reserved=8442.0MB
```
- Conclusion: Stage A invariants hold; proceed to Stage C pointer‑ring experiments.

Issue log: Stage C (pointer mode) lag spikes and faster decoherence
- Symptom:
  - Massive one-off spikes after enabling pointer mode with compile ON: `step2_s≈17–22 s` (with normal frames between), and decoder autotune bursts; later frames stabilize.
  - Visuals: noticeable faster decoherence versus roll; subjective quality regression.
- Hypothesis:
  - Compile retraces due to dynamic indices and conditional wrap branches in pointer assembly when compile is ON, causing long stalls.
  - Possible scale/KV misalignment or subtle off-by-one in pointer head/len1/len2 slicing leading to faster decoherence even when runs appear stable.
- Immediate mitigation:
  - Force pointer mode OFF under compile; fall back to roll (`OWL_COMPILE=1` → pointer disabled automatically), keeping pointer experiments to compile OFF only.
- Next diagnostics:
  - Add pointer diagnostics (under profiling) to print `head`, `len1`, `len2`, `wp`, and `filled` on `get/update_done` to verify index math and guarantee per-token scales are sliced identically to K/V.
  - A/B decoherence with roll vs pointer under compile OFF for a short fixed seed sequence; if pointer still decoheres faster, investigate scale alignment and tail-slice combination.

Result: disabling pointer under compile restores stability (block decoding context)
- Observation:
  - With `OWL_RING_MODE=pointer` but `OWL_COMPILE=1`, pointer mode is auto-disabled, and spikes vanish: attn ≈ 6–14 ms, decoder ≈ 8–9 ms, pipeline FPS ≈ 50–70 (scene-dependent), no decoherence.
- Interpretation (block decoding):
  - We decode in blocks (frame-sized token chunks), not token-by-token. The cache writes `new_len=tokens_per_frame` per update and `get()` logically appends the current block during the first step.
  - The pointer assembly must respect block boundaries and produce a strictly contiguous oldest→newest window for FlexAttention while ensuring the per-token scales for the exact block slices are used. Any wrap math error or data-dependent branching under compile can cause instability or visual drift.
- Action taken:
  - Keep roll-as-update as the stable default under compile; pointer mode is restricted to compile-OFF debugging until a compile-safe, fused gather is available.
- Future pointer plan (for block decoding):
  - Avoid dynamic branches under compile by always executing a fixed two-slice path with constant shapes; consider fusing timewise dequant + two-slice gather into a single kernel.
  - Validate that `get()` for the first sampling step (no update) returns `[filled + current_block]` with correct scale alignment; for the second step (with update), return `[filled]` only, matching mask semantics.

Latest pointer attempt (gather reorder under compile) – still blurry + spikes
- Command:
```bash
OWL_COMPILE=1 OWL_RING_POINTER_COMPILE=1 OWL_PROFILE_KV=1 OWL_FP8_KV=1 OWL_K_FP8=0 OWL_RING_MODE=pointer OWL_RING_GUARD=1 python -m inference.game_cv
```
- Outcome:
  - Visual decoherence: output mostly a single blurred color with faint UI elements.
  - Lag spikes persist under compile.
- Implications:
  - Even branchless gather with dynamic indices is causing poor kernel selection/retrace and/or KV/scale misalignment in practice.
  - We need a stricter compiler-safe design tailored to block decoding.

Root-cause hypotheses (decoherence + spikes)
- Decoherence (blurry, flat color):
  - Mismatch between K/V time reorder and per-token scales after gather (subtle off-by-one during head advance).
  - Inadvertent dtype/layout changes around gather; loss of contiguity before FlexAttention or incorrect broadcast in dequant.
  - Using partially filled windows early in prefill with pointer reorders.
- Spikes under compile:
  - Dynamic index content still induces graph recompilations or prevents fusions.
  - Gather on large tensors with non-trivial index patterns may force fallback codegen.

Compiler-safe pointer designs (block-decoding aware)
Option A: Roll-on-update (current stable default)
- Keep as primary path under compile; minimal risk, known stable.

Option B: Blockwise rotation via precomputed LUT (preferred)
- Assumptions: head advances by `tokens_per_frame` consistently. Let `B = T / tokens_per_frame` (number of blocks).
- Precompute `B` static index tensors that map `[0..T-1]` → rotated-by-`b*tokens_per_frame` order.
- At runtime, select the index tensor by phase `p = (offset / tokens_per_frame) % B` and gather K/V and scales using that fixed-shape index.
- Why compiler-friendly: shapes and control flow are constant; only tensor values change. Inductor can keep one graph where the index is a data tensor (no branching). Only `p` changes the data passed, not the graph.
- Advantages: exact block semantics, per-token scales rotate identically, no per-step roll/copy on update.
- Caveat: Requires careful construction of index for K/V `[B,H,T,D]` and scales `[B,H,T]`; must ensure `[B,H,T,D]` contiguity post-gather.

Option C: Full-T dequant then blockwise torch.roll on the block axis
- Reshape time to `[blocks, block_len] = [B, tokens_per_frame]`, perform `torch.roll(..., shifts=-phase, dims=block_dim)`, then reshape back.
- Because only `phase ∈ [0..B-1]` changes, Inductor may specialize on dynamic values but keep a single graph; however, historical behavior still shows retraces on dynamic roll shifts on some stacks.
- Lower preference than Option B.

Option D: Fused dequant+gather kernel (Triton)
- Single kernel to read int8 K/V and per-token scales from ring positions and write into a contiguous BF16 buffer.
- Most robust for performance; requires custom kernel development and careful integration.

Chosen path and implementation plan
1) Implement Option B (blockwise LUT gather) under pointer mode
   - Precompute once per cache: `idx_time[p]` for `p ∈ [0..B-1]` of shape `[1,1,T,1]` (for K/V) and `[1,1,T]` (for scales). Also cache an expanded `[1,1,T,D]` variant as needed.
   - At get(): compute `phase = (offset // tokens_per_frame) % B` (integer tensor, no `.item()`), pick `idx_time[phase]` and gather K/V and scales, then dequantize.
   - Always return `[B,H,filled,D]` contiguous BF16; if `filled < T`, slice tail.
   - Ensure no Python control flow on `phase`; pass phase-derived indices as tensors.

2) Strict alignment checks in debug
   - After gather, verify a tiny checksum vs a roll-based assembly on a few steps (compile OFF) to guarantee parity.
   - Log phase, head, and sample index triplets `(head, head+1, ..., head+block_len-1)` to confirm mapping.

3) Keep roll as default under compile until Option B proves stable
   - Gate with `OWL_RING_POINTER_COMPILE=1` for A/B; otherwise, fall back to roll.

4) If Option B still spikes, proceed to Option D
   - Prototype a Triton kernel for fused dequant+blockwise gather.

Validation plan (pointer, Option B)
- Unit:
  - For random tensors, compare pointer-get result vs roll assembly for several phases; assert exact equality for BF16 storage and acceptable tolerance for FP8 path.
- Integration:
  - Compile ON/OFF runs on two clips; track `attn_ms`, `decoder_ms`, `FPS`, `q_ms/dq_ms`, and visual quality. Expect no spikes and visuals matching roll.
- Gates:
  - Promote pointer under compile only if no spikes over 500+ frames and no visible decoherence.

Performance impact and priorities
- Observation: With compile ON, `q_ms/dq_ms≈0` and attention/decoder dominate. Switching roll ↔ pointer alone does not improve FPS.
- Highest-impact KV-side changes:
  - Fused dequant + blockwise gather (custom kernel): single pass to reorder and dequant K/V with per-token scales into contiguous BF16; compile-safe; minimizes extra BF16 traffic.
  - Tail-only reads aligned to block mask: assemble/dequantize only tokens actually consumed by FlexAttention’s local window. Combine with fusion for best effect.
- Additional lever (non-KV): Reduce local window size (L_local) if quality allows; directly reduces attention cost and typically outperforms bookkeeping changes.
- Low/neutral impact:
  - Pointer ring vs roll-on-update without fusion/tail-slicing is mainly semantic; expect negligible FPS change.

Next priorities
1) Implement Option B (blockwise LUT gather) for compile-safe pointer correctness.
2) Prototype Triton fused dequant+gather; measure end-to-end impact.
3) Add optional tail-only read path driven by the block mask; validate numerics.
4) Sweep L_local to identify acceptable quality/perf trade-offs.



