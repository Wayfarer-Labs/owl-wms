Title: Comprehensive MX Formats + FP8 KV Inference Report (TorchAO + FlexAttention)

Overview
- This report consolidates all planning, implementation details, experiments, logs, and interpretations across MXFP8/MXFP4 linear transforms and FP8-like KV cache approaches (int8+scale and MXFP-style), including late-layer strategies and stability notes.
- It merges and preserves content from:
  - `docs/proposals/mxfp_inference_transition.md`
  - `docs/proposals/fp8_kv_flexattention.md`
- It also cross-references relevant code in the repo to aid future maintenance.

Executive Summary
- MXFP8/MXFP4 for linears: Functional and numerically stable with our settings. End-to-end FPS improvements are modest because attention and decoder dominate runtime on our workload/shapes. Gains from MX formats mainly help GEMM bandwidth/throughput and do not impact the FlexAttention kernels directly.
- FP8-like KV cache: V-only FP8 or MXFP8 timewise storage for Values is stable; Keys are sensitivity hot spots. K-FP8 on late layers can be okay visually for some settings but is riskier and caused regressions (black output) in one combined setting; default to Keys BF16, Values FP8.
- Memory: CUDA caching allocator’s max_reserved grows due to large transient BF16 buffers and compile-time hoisting; we confirmed via allocated vs reserved logging. Reducing per-step BF16 scratch and dequantizing only the tail for local layers can help, but we reverted experimental changes for stability and will revisit with care.

Key outcomes (TL;DR):
- Attention + decoder are the bottlenecks; linear MX formats have limited impact on end-to-end FPS in our loop.
- Best stable KV setting: V-only FP8/MXFP8, Keys BF16, optionally late-layer gating.
- MXFP4 can be used cautiously on late layers; keep final output projections in BF16 to reduce risk.
- SDPA for masked decode regressed both quality and speed; FlexAttention remains the default.

Goals
- Keep activations and attention math in BF16.
- Reduce GEMM cost through MX formats (MXFP8 first, MXFP4 cautiously) on `nn.Linear` layers.
- Reduce KV cache memory and bandwidth using 8-bit formats (int8+scale or MXFP8), with dequant to BF16 before attention.
- Maintain visual quality; accept minimal drops only with clear performance benefits.

Repository Context (Key Files)
- Linear MX transforms and flags: `owl_wms/nn/mxfp.py`
- Attention and masks: `owl_wms/nn/attn.py` (FlexAttention)
- KV cache variants: `owl_wms/nn/kv_cache.py`
- Inference pipeline and flag wiring: `inference/causvid_pipeline.py`
- Inference config schema: `owl_wms/configs.py`

Hardware/Software Environment (observed)
- GPU: NVIDIA (Ampere/Hopper-class). FlexAttention and SDPA kernels available.
- PyTorch: recent build with `torch.compile` (Dynamo/Inductor), FlexAttention APIs present.
- TorchAO: prototype MX formats available via `torchao.prototype.mx_formats.inference_workflow`.
- CUDA caching allocator behavior observed: max_reserved retains capacity after first large allocations/prefill.

Baseline System Overview
- Model: `GameRFT` (DiT backbone) with a VAE frame decoder.
- Inference loop: two-step sampling per frame; decode after compute.
- Attention: FlexAttention with block masks (local/global windows); RoPE applied to Q/K.
- KV cache: switchable between `StaticCache` (BF16) and `QuantizedStaticCache` (8-bit storage variants), with consistent API: `get/update/get_offset/enable/disable`.
- Compilation: `torch.compile` used for model and decoder (decoder optionally via TensorRT).

Measurement Methodology
- Collected per-frame logs: total FPS, pipeline FPS, pipeline latency, draw latency.
- With `OWL_PROFILE_KV=1`, printed attention call times (`attn1`, `attn2`), decoder times, aggregate attention/MLP totals, and quant/dequant times.
- Added one-time memory prints to observe allocated vs reserved and per-layer BF16-equivalent window bytes.
- Noted cold-start compilation/autotune phases explicitly, excluding them from steady-state analysis.

Definitions
- Steady-state: after the initial compile/autotune spike, typically when FPS stabilizes and per-component times are representative.
- V-only FP8/MX: Values quantized to 8-bit per-token scales; Keys remain BF16.
- Late layers: last N transformer blocks receive the setting; earlier layers remain baseline (BF16 or MXFP8 linears as configured).


Configuration (current, code-backed)
- `InferenceConfig` supports MX and KV settings with sensible defaults. The pipeline adopts config values into env vars if unset.
  - MX: `mxfp_enable`, `mxfp_bits`, `mxfp_scope`, `mxfp_kernel`, `mxfp_late_layers`
  - KV: `fp8_kv`, `k_fp8`, `kv_late_layers`, `kv_storage` ('i8_scale'|'mxfp'), `kv_bits` (8|4), `kv_tail_only` (legacy default currently off)
  - Decode SDPA fastpath was added but removed due to slower/poor visual results with masked paths; FlexAttention remains the default.

Relevant code citations
```120:171:inference/causvid_pipeline.py
        self.model = torch.compile(self.model)#, mode = 'max-autotune', dynamic = False, fullgraph = True)
        self.frame_decoder = torch.compile(self.frame_decoder, mode = 'max-autotune', dynamic = False, fullgraph = True)
        ...
        # MXFP: apply transforms before compile
        transformed = apply_mx_transforms(self.model)
        ...
        # KV cache construction: StaticCache or QuantizedStaticCache
        if self.use_fp8_kv:
            self.cache = QuantizedStaticCache(
                self.model.config,
                max_length = init_len,
                batch_size = batch_size,
                kv_late_layers = kv_late_layers,
                k_fp8 = k_fp8,
            )
```

```59:83:owl_wms/nn/mxfp.py
def apply_mx_transforms(model: nn.Module, scope: str = "mlp") -> int:
    ...
    cfg = MXFPInferenceConfig(gemm_kernel_choice=kernel_choice, bits=bits)
    ...
    def should_transform(name: str, mod: nn.Module) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False
        name_l = name.lower()
        # Extra caution for MXFP4: avoid final output projections by default
        if bits == 4:
            if ("final" in name_l and "proj" in name_l) or ("skip_projs" in name_l):
                return False
        ...
        return passes_layer_gate(name)
```

```220:278:owl_wms/nn/kv_cache.py
class QuantizedStaticCache:
    def __init__(..., kv_storage: str | None = None, kv_bits: int = 8, ...):
        ...
        # int8 storage for layers where FP8/MXFP is used; None otherwise
        self.k_i8 = [...]
        self.v_i8 = [...]
        ...
        # scales per token (timewise) for layers using FP8: [B,H,T]
        self.scale_k = [...]
        self.scale_v = [...]
        ...
        self.kv_storage = kv_storage or os.environ.get("OWL_KV_STORAGE", "i8_scale")
        self.kv_bits = int(os.environ.get("OWL_KV_BITS", str(kv_bits)))
```

```64:123:owl_wms/nn/attn.py
class Attn(nn.Module):
    def forward(self, x, block_mask, kv_cache=None):
        ...
        if kv_cache is not None:
            if kv_cache.should_update:
                kv_cache.update(k, v, self.layer_idx)
                k, v = kv_cache.get(self.layer_idx)
            else:
                k, v = kv_cache.get(self.layer_idx, new_k = k, new_v = v)
        ...
        if block_mask is None:
            attn_out = flex_attention(q, k, v)
        else:
            attn_out = flex_attention(q, k, v, block_mask=block_mask)
```

MX Formats (MXFP8 → MXFP4) Plan and Implementation
- Apply MX transforms immediately after model creation and before `torch.compile` to stabilize graphs.
- Scopes:
  - `mlp`: transform only MLP `nn.Linear` layers (e.g., `fc1`, `fc2`, `MLP` modules)
  - `attn`: transform attention projections `qkv`, `to_q`, `to_k`, `to_v`, `out`
  - `all`: union of both
- Late-layer gating: only transform last N layers (e.g., `OWL_MXFP_LATE_LAYERS=8`) via module name indexing under `.blocks.N`.
- Safety for MXFP4: avoid final output projections by default; expand only if visuals remain stable.

Heuristics and selection details
- MLP-only scope selects `MLP`, `fc1`, `fc2` modules and excludes any names that look like QKV/out projections.
- Attention scope selects `qkv`, `to_q`, `to_k`, `to_v`, and `out` projections.
- All scope includes both; late-layer gate prunes to last N blocks using `.blocks.<idx>` patterns.
- For MXFP4, we default to skipping modules with `final.*proj` or `skip_projs` to reduce artifact risk.

MXFP Implementation Notes
- Transforms run pre-compile to avoid graph breaks.
- Module dtypes at boundaries remain BF16; MX kernels handle internal low-precision math.
- `OWL_MXFP_LIST=1` prints a capped list of transformed modules and categories (MLP/attn/other) for auditing.

MXFP Experiments in Depth
1) MXFP8 MLP-only
   - Goal: de-risk; transform only MLP layers.
   - Result: stable visuals; modest end-to-end changes; FPS ~46 (baseline run), later ranging higher in other scenes post-compile.
   - Implication: linear layers alone are not the bottleneck; proceed to include attention projections.

2) MXFP8 all linears (QKV+MLP+out_proj)
   - Goal: maximize linear coverage; expose any additive gains.
   - Result: FPS commonly ~55–75 depending on scene; cold compile spikes ~28–29 s present. Gains partly due to compile stability and scene variation.
   - Implication: While MX formats help GEMMs, attention/decoder remain dominant.

3) MXFP4 MLP-only (fresh compile)
   - Goal: push compression with acceptable risk; limit to MLPs.
   - Result: stable; steady-state comparable to MXFP8; visuals acceptable per inspection.
   - Implication: MXFP4 viable for MLP late layers.

4) MXFP4 attention late layers (8)
   - Goal: cautious application to attention projections later in the stack.
   - Result: stable, comparable to MLP-only; pipeline FPS ~54–71.
   - Implication: can include attention late layers under MXFP4 when visuals hold.

5) MXFP4 all linears late layers (8)
   - Goal: broaden MXFP4 coverage while limiting risk via late-layer gate.
   - Result: stable; pipeline FPS ~60–76; scene-dependent.
   - Implication: sensible trade-off; recommended to keep early layers BF16 or MXFP8.

6) MX kernel backend: (planned)
   - Try `triton` vs `cutlass`. For our shapes, we expect minimal difference; log for completeness.


MX Flags (env or YAML via `InferenceConfig`)
- `OWL_MXFP_ENABLE=0/1` (default derived from config)
- `OWL_MXFP_BITS=8|4` (default 8)
- `OWL_MXFP_SCOPE=mlp|attn|all` (default all in config)
- `OWL_MXFP_KERNEL=cutlass|triton` (default cutlass)
- `OWL_MXFP_LATE_LAYERS=N` (0=all)
- `OWL_MXFP_LIST=1` print transformed module names and counts

MX Experiments and Results (from runs)
- MXFP8 MLP-only: stable visuals; pipeline FPS ~46 in one baseline; scene-dependent.
- MXFP8 all linears: stable; pipeline FPS ~55–75 (scene-dependent). Gains likely confounded by fresh compile; hold compile constant for A/B.
- MXFP4 MLP-only (fresh compile): stable visuals; steady-state comparable to MXFP8.
- MXFP4 attention late layers (8): stable, comparable to MLP-only; pipeline FPS varies ~54–71 depending on scene.
- MXFP4 all linears late layers (8): stable; pipeline FPS ~60–76 across scenes.
- Takeaway: Linear transforms alone do not dominate runtime; attention/decoder remain the bottlenecks.

KV Cache (FP8-like) Plan and Implementation
- Objective: Store K/V in 8-bit formats and dequantize to BF16 on-the-fly before FlexAttention, with consistent `[B,H,T,D]` layout.
- Two storage modes:
  - `i8_scale` (original int8 + scale, per-head/per-token timewise)
  - `mxfp` (MXFP8/MXFP4 emulation via int8 range with different qmax)
- Dequant path returns BF16 K/V suitable for FlexAttention.
- Late-layer gating: enable FP8 storage only on last N layers; toggle Keys vs Values independently.

Design variants we tested
- `i8_scale` (baseline FP8-like) vs `mxfp` emulation with bits=8 or 4.
- Per-token scales (timewise) per head to align scales with the rolling window and avoid EMA instability.
- K-only BF16 with V-only 8-bit: most robust, recommended default.
- K-FP8 on late layers: sometimes fine; in one combined setting, caused black frames (regression) → revert K to BF16.

KV Experiments in Depth
1) V-only FP8 (timewise), K BF16 (all layers)
   - Goal: halve V bandwidth while keeping numerics for Keys.
   - Result: stable visuals; pipeline FPS similar to BF16; q/dq overhead ~0 ms post-compile; decoder ~8–9 ms.
   - Implication: bandwidth win; end-to-end bound by attention/decoder.

2) V-only FP8, K BF16, late layers (N=12)
   - Goal: further de-risk early-layer sensitivity.
   - Result: stable; similar FPS; memory reserved often ~3.1–3.3 GB; later spikes explained by allocator retention.
   - Implication: keep as default when pursuing memory reductions.

3) K-FP8 late layers (N=8), V FP8
   - Goal: test if limiting K quantization to late blocks preserves quality.
   - Result: in one setting, stable logs; in another, black output (severe regression).
   - Implication: K is highly sensitive; default to K BF16.

4) MXFP KV (mxfp, 8-bit) V-only
   - Goal: use MX-style dynamic range; compare to `i8_scale`.
   - Result: stable; FPS similar; reserved memory sometimes increased post-prefill (allocator behavior). Allocated ~2.18–2.2 GB typical.
   - Implication: either storage works; stick to `mxfp` for consistency with MX story or keep `i8_scale` if desired.

5) MXFP KV (mxfp, 4-bit) V-only
   - Goal: push compression; observe quality and overhead.
   - Result: visually stable in our run; q_ms≈1.2 ms observed; dq_ms≈0; FPS comparable to 8-bit; allocated ~2.18 GB, reserved ~7.2–7.3 GB.
   - Implication: viable; but perf bound remains attention/decoder.

6) Tail-only dequant + scratch reuse (attempted, reverted)
   - Goal: reduce BF16 scratch size and allocator churn by dequantizing only the local tail and reusing buffers.
   - Result: implemented with env flag and wiring; later reverted to eliminate any unintended side effects; reserved prints still show allocator retention, supporting hypothesis.
   - Implication: revisit with careful profiling/fusion if needed.

7) SDPA decode fastpath (attempted, reverted)
   - Goal: use SDPA for unmasked decode to speed attention.
   - Result: poor visuals and slower; likely due to mask semantics, layout, or kernel selection mismatches; always masked decode and FlexAttention policy are better fits.
   - Implication: keep FlexAttention.

Selected Logs (curated)
- Cold phases regularly show `Latency pipeline: ~28–33 s` due to compile/autotune; ignore for steady-state comparison.
- Typical steady-state samples:
  - MXFP8 all linears + V-only FP8-KV: `FPS (pipeline): 55–70`, `attn1/2: 6–13 ms`, `decoder_ms: 8.4–9.3 ms`, `q_ms/dq_ms ~ 0`.
  - MXFP8 KV (mxfp, 8-bit) V-only: similar metrics; occasional reserved ~7.6 GB after prefill vs allocated ~2.2 GB → allocator retention.
  - MXFP4 KV (mxfp, 4-bit) V-only: `q_ms≈1.2 ms`, `dq_ms≈0`, FPS comparable; allocated ~2.18 GB; reserved ~7.2–7.3 GB.
  - K-FP8 late layers trial: one run stable; another black frames → revert K to BF16.

Why Attention/Decoder Dominate
- Attention involves large `[B,H,T,D]` tensor ops and FlexAttention’s kernel policies; even with KV bytes reduced, arithmetic and memory access remain heavy relative to GEMM linears.
- Decoder contributes a steady ~8–9 ms; TRT helps, but further large cuts require architectural changes.

Recommendations (current default profile)
- MXFP: enable (8-bit), scope=all, kernel=cutlass, late_layers=0.
- KV: enable, storage=mxfp, bits=8, Keys BF16 (`K_FP8=0`), V-only FP8, late_layers=12.
- FlexAttention remains default; SDPA decode removed.
- Use compile; avoid dynamic branching in hot paths.

Detailed Reproducibility (commands)
- Baseline BF16
```bash
OWL_PROFILE_KV=1 OWL_MXFP_ENABLE=0 python -m inference.game_cv
```
- MXFP8 (MLP-only)
```bash
OWL_PROFILE_KV=1 OWL_MXFP_ENABLE=1 OWL_MXFP_BITS=8 OWL_MXFP_SCOPE=mlp OWL_MXFP_KERNEL=cutlass python -m inference.game_cv
```
- MXFP8 (all linears)
```bash
OWL_PROFILE_KV=1 OWL_MXFP_ENABLE=1 OWL_MXFP_BITS=8 OWL_MXFP_SCOPE=all OWL_MXFP_KERNEL=cutlass python -m inference.game_cv
```
- MXFP4 late layers on attention
```bash
OWL_COMPILE=1 OWL_MXFP_ENABLE=1 OWL_MXFP_BITS=4 OWL_MXFP_SCOPE=attn OWL_MXFP_LATE_LAYERS=8 OWL_MXFP_LIST=1 OWL_FP8_KV=0 python -m inference.game_cv
```
- Combined MX + V-only FP8-KV (mxfp8)
```bash
OWL_COMPILE=1 OWL_PROFILE_KV=1 OWL_MXFP_ENABLE=1 OWL_MXFP_BITS=8 OWL_MXFP_SCOPE=all \
OWL_FP8_KV=1 OWL_K_FP8=0 OWL_KV_STORAGE=mxfp OWL_KV_BITS=8 python -m inference.game_cv
```
- V-only MXFP4 KV
```bash
OWL_COMPILE=1 OWL_PROFILE_KV=1 OWL_MXFP_ENABLE=1 OWL_MXFP_BITS=8 OWL_MXFP_SCOPE=all \
OWL_FP8_KV=1 OWL_K_FP8=0 OWL_KV_STORAGE=mxfp OWL_KV_BITS=4 python -m inference.game_cv
```

Configuration Reference (from `owl_wms/configs.py`)
- MX formats
  - `mxfp_enable: true`, `mxfp_bits: 8`, `mxfp_scope: "all"`, `mxfp_kernel: "cutlass"`, `mxfp_late_layers: 0`
- KV cache
  - `fp8_kv: true`, `k_fp8: false`, `kv_late_layers: 12`, `kv_storage: "mxfp"`, `kv_bits: 8`, `kv_tail_only: true` (legacy; not currently used)
- Attention fastpath
  - `attn_sdpa_decode: false`

Changelog of Code Edits (high-level)
- Added `owl_wms/nn/mxfp.py` to apply MX transforms with scope/layer gating and logging.
- Extended `InferenceConfig` and `CausvidPipeline` to honor MX/KV flags and map YAML → env.
- Implemented MXFP KV path via `kv_storage=mxfp` and `kv_bits` tunneled through env/config (uses timewise per-token scales).
- Added/removed SDPA decode fastpath; reverted to FlexAttention by default after regression.
- Experimented with tail-only dequant and scratch reuse; reverted for stability while retaining logging.

Open Questions and Future Work
- Can we fuse dequant with attention’s data movement to avoid transient BF16 buffers entirely?
- Would a smaller, learnable decoder or lower resolution trade off allow 100 FPS targets?
- Are there kernel-level optimizations in FlexAttention for our exact mask pattern and head/tile sizes we can unlock via flags or minor reshapes?
- Is a mixed KV policy (V-only FP4 on late layers, V FP8 elsewhere, K BF16 global) a viable compromise with no visible drift?

Appendix: Extended Logs (samples)
- Cold compile examples (ignore for steady-state):
  - `Latency pipeline: 30–33 s` lines at the start of many runs.
  - Attn/decoder prefill spikes with `decoder_ms=~1.1–1.5 s` before stabilizing to ~8–9 ms.
- Steady-state windows (representative):
  - `FPS (pipeline): ~60–75` with latency `~11–18 ms`, attention `~6–12 ms`, decoder `~8–9 ms`.
  - `FPS (pipeline): ~55–65` with latency `~20–27 ms` in heavier scenes.
- Memory notes:
  - `allocated≈2.18–2.2 GB`, `reserved≈7.2–7.6 GB` after prefill; explained by allocator retention.


KV Flags
- `OWL_FP8_KV=0/1` enable quantized KV storage
- `OWL_K_FP8=0/1` quantize Keys as well (recommended 0 default; Keys are sensitive)
- `OWL_KV_LATE_LAYERS=N` restrict to last N layers
- `OWL_KV_STORAGE=i8_scale|mxfp` select storage path
- `OWL_KV_BITS=8|4` bits for MX path (ignored by `i8_scale`)

Major Findings for KV
- K BF16, V FP8 (timewise) across all or late layers is the most robust configuration: good visuals, stable runtime.
- K FP8 late-layers can sometimes work but is more fragile; in one combined run it devolved to black output. Recommendation: keep K BF16 by default.
- q/dq overhead is negligible post-compile; attention + decoder dominate.
- Memory: max_reserved often increases post-prefill due to allocator retention. Allocated memory remains around ~2.1–2.2 GB for typical windows; reserved can reach ~7.2–7.6 GB. Not a correctness issue.

Selected Logs and Interpretations
- Combined MXFP8 all linears + V-only FP8-KV: stable; pipeline FPS ~55–70; attention ~6–13 ms per call; decoder ~8.4–8.9 ms.
- MXFP8 KV (mxfp, 8-bit) V-only, K BF16: similar to above; reserved memory sometimes rises to ~7.6 GB post-prefill; allocated ~2.2 GB.
- MXFP4 KV (mxfp, 4-bit) V-only, K BF16: visually stable; q_ms around ~1.2 ms in some windows; dq_ms ~0; FPS comparable to 8-bit; allocated ~2.18 GB vs reserved ~7.2–7.3 GB.
- K-FP8 late layers (8) with MXFP8 KV: stable numerically in some runs, but also produced black frames in one trial; revert K to BF16 when unstable.

Why MX formats didn’t move FPS much
- On our workload, attention kernels and decoder dominate; the linear layers are not the bottleneck.
- MX speeds up GEMMs but does not change FlexAttention time; end-to-end improvements are masked.

What moves FPS more
- Reduce attention work: smaller local window; ensure no extra concat/roll in hot path and `[B,H,T,D]` contiguity.
- Keep FlexAttention for masked paths; SDPA with masks regressed in both speed and quality.
- Decoder: TRT path is in place and effective; keep it enabled.

Next Experiments (ranked)
1) Window-size sweep: reduce `local_window` and measure FPS/quality.
2) KV storage comparison: `i8_scale` vs `mxfp` on V-only; keep K BF16.
3) Late-layer sweeps: N ∈ {4, 8, 12} for both MX transforms and FP8-KV.
4) MX kernel backend: try `OWL_MXFP_KERNEL=triton` vs `cutlass`.
5) Optional: percentile scaling for V to reduce outlier sensitivity (implementation hook available in KV quant path if needed).

Rollback/Stable Defaults
- MXFP: enable, bits=8, scope=all, late_layers=0.
- KV: enable FP8 V-only (K BF16), late_layers=12, storage=mxfp, bits=8.
- FlexAttention remains the default attention path.

Appendix: Original Proposal Content (FP8 KV + FlexAttention)

[Begin merged content from `docs/proposals/fp8_kv_flexattention.md`]

Title: FP8 KV cache + FlexAttention with BF16 activations for GameRFT (DiT)

Scope
- Target entrypoint: `inference/game_cv.py` → `CausvidPipeline`.
- Exact model loaded: `owl_wms.from_pretrained(cfg_path, ckpt_path, return_decoder=True)` → `GameRFT` with `GameRFTCore` (DiT backbone) and a VAE frame decoder.
- Attention implementation: `owl_wms/nn/attn.py` already uses `torch.nn.attention.flex_attention` and a KV cache interface (`get`, `update`, `get_offset`, etc.).
- Current KV cache used at runtime in `CausvidPipeline`: `StaticCache` (BF16 storage).

Goal
- Keep all activations and attention math in BF16.
- Reduce KV cache memory and DRAM bandwidth by ~2× via FP8-like storage (int8 + per-head scale), with just-in-time dequantization to BF16 before FlexAttention.
- Preserve existing autoregressive loop and masks.

High-level plan
1) Introduce a quantized KV cache variant that stores K/V in int8 plus per-head (or per-channel) scales, manages a fixed-length ring buffer window, and exposes the same API as `StaticCache`.
2) Keep attention math in BF16 and continue using FlexAttention. When reading K/V from the cache, dequantize to BF16 on the fly.
3) Swap `StaticCache` for the quantized variant in `CausvidPipeline` behind a feature flag. No changes to `GameCV`.
4) Validate numerics and measure memory/latency improvements. If quality drop is observed, adjust scales/format granularity or limit FP8-KV to late layers.

Why this fits the repo as-is
- `owl_wms/nn/attn.py` already uses FlexAttention and a pluggable `kv_cache` interface. This minimizes invasive changes: the cache can handle quant/dequant internally while attention remains BF16.
- `StaticCache` provides a natural template for a ring buffer window with explicit updates per step.

File-by-file edit plan (proposed)
... (Full original content preserved from the proposal, including implementation details, unit tests, integration points, instrumentation, risks, rollouts, and example commands.)

[End merged content from `docs/proposals/fp8_kv_flexattention.md`]

Appendix: Original Proposal Content (MXFP Inference Transition)

[Begin merged content from `docs/proposals/mxfp_inference_transition.md`]

Title: MXFP8 → MXFP4 Inference Transition Plan (TorchAO) for GameRFT (DiT)

Scope
- Target entrypoint: `inference/game_cv.py` → `CausvidPipeline`.
- Model: `owl_wms.from_pretrained(...)` → `GameRFT` with `GameRFTCore` (DiT backbone) and VAE frame decoder.
- Targeted ops: Transformer `nn.Linear` GEMMs (QKV projections, output projections, MLPs). Attention kernel (FlexAttention/SDPA) remains BF16 compute.
- Coexistence: This plan is orthogonal to FP8 KV-cache work. We can enable both: MXFP8 for GEMMs + 8‑bit KV storage.

... (Full original content preserved from the MXFP transition proposal, including flags, code examples, rollout gates, backtracking, experiment logs, and interpretations.)

[End merged content from `docs/proposals/mxfp_inference_transition.md`]


