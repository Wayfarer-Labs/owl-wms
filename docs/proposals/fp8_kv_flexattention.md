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
- `owl_wms/nn/attn.py` already uses FlexAttention and a pluggable `kv_cache` interface. This minimizes invasive changes: the cache can handle quantization/dequantization internally while attention remains BF16.
- `StaticCache` provides a natural template for a ring buffer window with explicit updates per step.

File-by-file edit plan (proposed)

1) `owl_wms/nn/kv_cache.py`
- Add a new class `QuantizedStaticCache` with the same public methods used by attention:
  - `enable_cache_updates()`, `disable_cache_updates()`
  - `get(layer_ind, new_k=None, new_v=None)`
  - `update(new_k, new_v, layer_ind)`
  - `get_offset(idx)`, `reset(batch_size)`, `truncate(...)` (no-op), `shape`, `to(...)`

- Storage layout per layer (int8 plus scales):
  - `k_i8[layer]`: `[B, H, max_tokens, D]` int8
  - `v_i8[layer]`: `[B, H, max_tokens, D]` int8
  - `scale_k[layer]`: `[B, H]` BF16/FP32 (per-head) or `[B, H, D]` (per-channel)
  - `scale_v[layer]`: same as above
  - Offsets and a ring write pointer. `max_tokens = max_length * tokens_per_frame` (match `StaticCache`).

- Write path (update/append):
  - Given BF16 `new_k/new_v` of shape `[B, H, T_new, D]`, compute amax over `(T_new, D)` per head.
  - EMA-update amax and derive scale per head. For initial rollout: K in E5M2-like (bigger exponent), V in E4M3-like.
  - Quantize: `q = clamp(round(x / scale), -128..127).to(int8)`.
  - Roll the window or advance ring pointer and write the int8 blocks. Update `offsets[layer_ind] += T_new`.

- Read path (get):
  - If `new_k/new_v` are provided, logically return the view that would result from appending them (do not mutate storage yet). This mirrors `StaticCache.get` behavior used in attention.
  - Otherwise, return the full current window by dequantizing the int8 slab with stored scales into a temporary BF16 tensor of shape `[B, H, T_win, D]`.

- Public API must match existing usage patterns in `owl_wms/nn/attn.py::Attn.forward`:
```python
if kv_cache is not None:
    if kv_cache.should_update:
        kv_cache.update(k, v, self.layer_idx)        # quantize+write here
        k, v = kv_cache.get(self.layer_idx)          # returns BF16 window
    else:
        k, v = kv_cache.get(self.layer_idx, new_k=k, new_v=v)  # logical BF16 concat
```

2) `owl_wms/quant/fp8_kv.py` (new)
- Add quantization helpers (int8 + scale) with EMA:
```python
import torch

@torch.no_grad()
def ema_update(prev: torch.Tensor, new: torch.Tensor, momentum: float = 0.99) -> torch.Tensor:
    return momentum * prev + (1.0 - momentum) * new

@torch.no_grad()
def quantize_per_head(x_bf16: torch.Tensor, amax_prev: torch.Tensor, fmt: str = "e4m3"):
    # x: [B,H,T,D] or [H,T,D] (treat B=1)
    if x_bf16.dim() == 4:
        x = x_bf16[:, :, :, :]
        reduce_dims = (2, 3)
    else:
        x = x_bf16
        reduce_dims = (1, 2)
    amax_now = x.abs().amax(dim=reduce_dims)
    amax = ema_update(amax_prev, amax_now)
    denom = 127.0
    scale = (amax / denom).clamp_min(1e-8)
    inv = (1.0 / scale).view(*scale.shape, 1, 1)
    if x_bf16.dim() == 4:
        q = (x_bf16 * inv[:, :, None, None]).round().clamp_(-128, 127).to(torch.int8)
    else:
        q = (x_bf16 * inv[:, None, None]).round().clamp_(-128, 127).to(torch.int8)
    return q, scale, amax

@torch.no_grad()
def dequantize_per_head(q_i8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if q_i8.dim() == 4:  # [B,H,T,D]
        return (q_i8.float() * scale[:, :, None, None]).to(torch.bfloat16)
    else:  # [H,T,D]
        return (q_i8.float() * scale[:, None, None]).to(torch.bfloat16)
```

- Optional per-channel variants if needed: same functions with `scale` of shape `[B,H,D]` or `[H,D]`.

3) `owl_wms/nn/attn.py`
- Minimal changes: keep using FlexAttention (already present). Ensure that cached K/V returned by `kv_cache.get(...)` are BF16 and in `[B,H,T,D]` layout as currently expected by FlexAttention. Today the code does:
```python
qkv = self.qkv(x)
q, k, v = einops.rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=self.n_heads)
...
if kv_cache is not None:
    if kv_cache.should_update:
        kv_cache.update(k, v, self.layer_idx)
        k, v = kv_cache.get(self.layer_idx)
    else:
        k, v = kv_cache.get(self.layer_idx, new_k = k, new_v = v)
...
attn_out = flex_attention(q, k, v, block_mask=block_mask if block_mask is not None else None)
```
- The quantized cache will uphold this contract by returning BF16 K/V; FlexAttention remains unchanged.

Self-criticism: If we discover the cache needs a distinct fast path for dequantizing only the last N tokens, we can add an API like `get_slice(layer, start, end)` to reduce dequantization volume. Start simple with full-window dequant.

4) `inference/causvid_pipeline.py`
- Add a feature flag (env/config) `use_fp8_kv`. When true, construct `QuantizedStaticCache` in `_build_cache()` instead of `StaticCache`.
- No change to the two-step diffusion logic; the model calls stay the same. Prefill still runs once with updates enabled; decode runs with a read, then a read+update.

Sketch:
```python
self.cache = QuantizedStaticCache(self.model.config, max_length=init_len // self.model.config.tokens_per_frame, batch_size=1)
...
self.cache.enable_cache_updates()
_ = self.model(prev_x_noisy, prev_t, self.mouse_buffer, self.button_buffer, kv_cache=self.cache)
self.cache.disable_cache_updates()
```

Self-criticism: `StaticCache` uses absolute token counts internally. Ensure that `max_length * tokens_per_frame` matches your `init_len` tokens and that ring updates maintain `offsets[layer]` semantics used by RoPE in attention.

5) `inference/game_cv.py`
- No functional change. Optionally print whether FP8-KV mode is enabled.

Config and flags
- Add to your config or accept env vars:
  - `OWL_FP8_KV=1` to enable quantized cache
  - `OWL_KV_FMT_K=e5m2`, `OWL_KV_FMT_V=e4m3`
  - `OWL_KV_SCALE=head` (or `channel`)
  - `OWL_KV_EMA=0.99`

Validation plan
1) Unit tests
- Quant/dequant round-trip error per head for synthetic `[H,T,D]` tensors.
- Ring buffer wrap-around writes and reads.

2) Integration
- Instrument attention to dump a small histogram of K/V amax/scale per layer to spot saturation.
- Compare `pred_v` vs baseline (BF16 cache) for 100 decode steps from the same initialization.

3) E2E metrics
- PSNR/SSIM of decoded frames for 2–3 short rollouts.
- Memory: `torch.cuda.memory_allocated()` before/after prefill.
- Latency breakdown: time spent in `kv_cache.get`/`update` and FlexAttention.

Backtracking guide (self-critique)
- If accuracy drops: switch K to E5M2-like scaling while keeping V in E4M3. If still problematic, use per-channel scales for K only.
- If performance regresses: check that layouts are contiguous before FlexAttention; dequant into a contiguous BF16 tensor. Avoid Python-side slicing in hot loops.
- If FlexAttention underperforms on your driver: temporarily use SDPA (`F.scaled_dot_product_attention`) and keep FP8-KV storage; bandwidth savings still apply.
- If `torch.compile` retraces too often: keep quant/dequant helpers simple and annotate any dynamic branches; test without compile first.

Risks
- Additional compute for dequantization per step. Mitigated by large DRAM savings and small T (window length ~60).
- Scale instability at start of session. Warm up with a few EMA steps or clamp amax growth.
- Mismatch between `offsets` and ring pointer can break RoPE; unit-test offsets across updates.

Rollout phases
- Phase A: Enable FP8-KV on last N layers only; measure drift and FPS.
- Phase B: All layers per-head scales.
- Phase C: Per-channel scales for K only if required.
- Phase D (optional): Migrate dequant into fused kernels (TransformerEngine) once stable.

Appendix: Minimal `QuantizedStaticCache` skeleton (conceptual)
```python
class QuantizedStaticCache:
    def __init__(self, config, max_length, batch_size=1, device='cuda', dtype=torch.bfloat16,
                 fmt_k='e5m2', fmt_v='e4m3', ema_m=0.99, per_channel=False):
        T = max_length * config.tokens_per_frame
        H = config.n_heads
        D = config.d_model // H
        B = batch_size
        self.k_i8 = [torch.empty(B, H, T, D, device=device, dtype=torch.int8) for _ in range(config.n_layers)]
        self.v_i8 = [torch.empty(B, H, T, D, device=device, dtype=torch.int8) for _ in range(config.n_layers)]
        self.scale_k = [torch.ones(B, H, device=device, dtype=torch.bfloat16) for _ in range(config.n_layers)]
        self.scale_v = [torch.ones(B, H, device=device, dtype=torch.bfloat16) for _ in range(config.n_layers)]
        self.offsets = [0] * config.n_layers
        self.tokens_per_frame = config.tokens_per_frame
        self.should_update = False
        self.per_channel = per_channel
        self.fmt_k, self.fmt_v, self.ema_m = fmt_k, fmt_v, ema_m

    def enable_cache_updates(self):
        self.should_update = True

    def disable_cache_updates(self):
        self.should_update = False

    def get_offset(self, idx=0):
        return self.offsets[idx]

    @torch.no_grad()
    def get(self, layer_ind, new_k=None, new_v=None):
        # return BF16 tensors [B,H,T,D]; if new_* provided, logically append
        k = dequantize_per_head(self.k_i8[layer_ind][0], self.scale_k[layer_ind][0]).unsqueeze(0)
        v = dequantize_per_head(self.v_i8[layer_ind][0], self.scale_v[layer_ind][0]).unsqueeze(0)
        if new_k is not None:
            k = torch.cat([k, new_k], dim=2)
            v = torch.cat([v, new_v], dim=2)
        return k, v

    @torch.no_grad()
    def update(self, new_k, new_v, layer_ind):
        # quantize and write last positions into ring
        B, H, T_new, D = new_k.shape
        qk, sk, ak = quantize_per_head(new_k[0], self.scale_k[layer_ind][0], self.fmt_k)
        qv, sv, av = quantize_per_head(new_v[0], self.scale_v[layer_ind][0], self.fmt_v)
        # roll-like update (write to end)
        self.k_i8[layer_ind] = torch.roll(self.k_i8[layer_ind], shifts=-T_new, dims=2)
        self.v_i8[layer_ind] = torch.roll(self.v_i8[layer_ind], shifts=-T_new, dims=2)
        self.k_i8[layer_ind][0, :, -T_new:, :] = qk
        self.v_i8[layer_ind][0, :, -T_new:, :] = qv
        self.scale_k[layer_ind][0] = sk
        self.scale_v[layer_ind][0] = sv
        self.offsets[layer_ind] += T_new
```

Reality check versus your exact stack
- Model: `GameRFTCore` → `DiT` → `Attn` already uses FlexAttention and a `kv_cache` with the same read/write hooks we need. Great.
- Pipeline: `CausvidPipeline` uses `StaticCache` and toggles updates exactly where we’d quantize/store. Swapping to `QuantizedStaticCache` is straightforward.
- Activations: Pipeline and model cast to BF16; we keep it that way.

What success looks like
- Memory: `n_layers × B × H × T × D × 1 byte` for K plus V, instead of `2 bytes`, plus tiny scale overhead. Expect ~2× KV memory reduction.
- Latency: Reduced DRAM bytes per decode step; slight compute increase for dequant but overall equal or faster.
- Quality: PSNR/SSIM near baseline; if not, apply the backtracking steps.

Self-criticism and backtracking (expanded)
- Dequant compute overhead: We are adding per-step dequantization (int8→BF16) for full KV windows per layer. While arithmetic cost is small relative to DRAM savings, it is not zero. If profiling shows dequant hot, mitigate by:
  - Pre-allocating contiguous BF16 scratch buffers and reusing them (avoid allocs).
  - Fusing dequant with a layout transform so FlexAttention receives a contiguous, head-major view directly.
  - Adding an API to dequantize only the slice needed (if mask restricts reads to a strict local window smaller than `WINDOW_SIZE`).

- Coarseness of per-head scaling: Per-head scales can be too blunt, especially for Keys which often have broader dynamic range. If drift appears:
  - Switch Keys to E5M2-like (more exponent headroom), keep Values in E4M3.
  - Escalate Keys (only) to per-channel scaling (`[H,D]`). Values can usually remain per-head.
  - As a last resort, keep Keys in BF16 and Values in FP8 (still halves V bandwidth and preserves numerics of K).

- Scale stability (EMA warm-up): Early in a session, EMA estimates can undershoot or overshoot, causing flicker. We should:
  - Initialize amax with a small calibration pass (few frames) using running maxima.
  - Clamp EMA growth factor and enforce a minimum scale floor.
  - Optionally decouple EMA for K and V (different momenta).

- Ring buffer and RoPE offsets: Any mismatch between ring pointer, logical token order, and `get_offset()` will break positional encoding. To prevent this:
  - Unit-test `offsets` monotonicity and equality with effective KV length per layer after each update.
  - Assert that `get(layer, new_k, new_v)` returns a view consistent with the `update()` state transition.
  - Add a checksum mode that compares a BF16 `StaticCache` vs `QuantizedStaticCache` for a few steps.

- Mask parity with FlexAttention: Our current block/causal policy must be bitwise equivalent. Risks include off-by-one at frame boundaries and window edges. Actions:
  - Generate dense masks for small toy sequences with both methods and compare.
  - Add a debug mode to dump a few attention score rows to ensure equivalence.

- Interaction with `torch.compile`: Quant/dequant helpers add ops to the graph. If recompiles occur frequently or fusion degrades:
  - Keep helpers simple, branchless, and mark them compatible with Dynamo.
  - Prototype without `torch.compile` first; enable once shapes are steady.
  - If needed, place helpers behind `torch._dynamo.allow_in_graph`-friendly code paths.

- Memory fragmentation: Repeatedly allocating new dequant buffers can fragment memory. Pre-allocate per-layer scratch BF16 buffers sized to the window and reuse them.

- Measurement bias: Improvements may be masked by display overhead in `game_cv`. Always report pipeline-only latency (already printed) and, for development, run headless benchmarks.

Risks and mitigations (expanded)
- Risk: Quality regression with FP8-KV
  - Mitigation: K→E5M2, V→E4M3; per-channel scales for K; limit FP8-KV to later layers first; keep K in BF16 if necessary.

- Risk: Performance regression due to dequant
  - Mitigation: Pre-allocate contiguous scratch; avoid Python slicing; ensure tensors are in `[B,H,T,D]` contiguous layout; profile to confirm DRAM bytes halved.

- Risk: FlexAttention fallback to slower path
  - Mitigation: Validate backend selection; if needed, temporarily use SDPA with the same mask; bandwidth win from FP8-KV remains.

- Risk: Offset/mask mismatch
  - Mitigation: Add unit tests that assert identical outputs for small sequences between BF16 `StaticCache` and `QuantizedStaticCache`.

- Risk: Scale overflow/underflow
  - Mitigation: Clamp amax, impose scale floors/ceilings, track saturation counters per head.

Decision checkpoints and go/no-go gates
- Checkpoint 1 (Last N layers only): If PSNR drop > 0.5 dB or noticeable artifacts, escalate K scaling granularity and/or switch K to E5M2.
- Checkpoint 2 (All layers): If additional drop occurs, revert early layers to BF16 or per-channel K only.
- Checkpoint 3 (Throughput): If dequant dominates, optimize memory layout and scratch buffers; if still slow, consider TE to fuse dequant.

Failure modes and instrumentation
- Add counters: total dequant bytes, saturation events (|q| = 127), EMA convergence rate.
- Log min/mean/max of K/V scales per layer per 100 steps.
- Optional sanity: run a short decode with both caches in parallel and compare `pred_v` L2 every N steps.

Experiment design (A/B)
- A: Baseline BF16 `StaticCache`.
- B1: FP8-KV on last 1/3 layers, per-head scales (K=E5M2, V=E4M3).
- B2: FP8-KV on all layers, per-head.
- B3: FP8-KV on all layers, per-channel K only.
- Metrics: pipeline latency, memory, PSNR/SSIM, `pred_v` L2 drift.

Performance model (rough)
Incident log (errors encountered and resolutions)

1) Runtime failure: "expand: attempting to expand a dimension of length 3840!" during prefill
- Symptom: Crash when writing into the quantized cache in `QuantizedStaticCache.update`, showing a zero-length time dimension and failing setitem into `[:, :, -new_len:, :]`.
- Root cause: Mismatch in interpretation of `max_length`. The quantized cache initially allocated `T = max_length` tokens assuming the caller passed tokens, but `CausvidPipeline` passed frames. Resulting T=init_len (frames) instead of tokens, leaving an effective zero-size when rolling/writing.
- Fix: Align semantics with `StaticCache` by allocating tokens as `T = max_length * tokens_per_frame`. In the pipeline, pass `max_length = init_len` frames, and let the cache multiply to tokens.
- Additional guard: In `update`, handle the initial fill case by avoiding `roll` when buffer is empty and clipping writes to buffer size.
- Prevention: Document and enforce that cache constructors take frames and internally convert to tokens; add asserts in debug builds to verify `T % tokens_per_frame == 0`.

2) Minor dtype choice in dequant (BF16 vs FP32)
- Symptom: Concern about dequant multiply using FP32 internally.
- Resolution: Keep dequant output in BF16 and optionally do the multiply in BF16 to satisfy the BF16-activations constraint. If drift appears on tiny scales, selectively upcast only the multiply, not the outputs.
- Prevention: Expose a flag to choose dequant multiply dtype and validate numerics over a few steps.

3) Visual corruption (output looks like noise), no FPS gain
- Symptom: Frames degenerate into noise; latency unchanged.
- Likely causes:
  - Scale misalignment across time when using a single per-head scale with EMA during early steps.
  - K/V scales not aligned to ring writes; dequant using wrong scales for recent tokens.
  - Overly aggressive quantization on Keys.
- Fix applied:
  - Switched to per-head, per-token (timewise) scaling: compute scale per [B,H,T] by amax over D, store scales alongside tokens in the ring buffer. This guarantees correct alignment and removes EMA instability during boot.
  - Dequant now uses matching per-token scales; Keys/Values remain BF16 in compute.
  - Retained K=E5M2 vs V=E4M3 option (format naming in code is advisory since it’s int8+scale), with per-token scales dominating quality.
- Prevention:
  - When introducing quantized KV for AR decoding, prefer per-token scales initially; only collapse to per-head EMA after stability is proven.
  - Add an assertion that scales slice indices match KV write indices.

Current observations (post-fix)
- Visual quality: Looks correct ~80% of the time but decoheres faster than BF16 baseline over longer rollouts.
- Throughput: No measurable FPS improvement in the pipeline, despite halving KV DRAM bytes in theory.

## Strengths
- Memory footprint for KV is halved; GPU memory headroom improved.
- Visual quality largely preserved short-term with per-token scaling.
- Integration overhead is modest: FlexAttention path unchanged; all quant logic isolated in cache.

## Weaknesses / Open questions
- Decoherence over time: residual drift indicates quantization noise (likely in Keys) accumulates across steps.
- No FPS gain observed: bandwidth reduction didn’t translate to lower step time; suggests compute or other memory (non-KV) dominates, or dequant/layout overhead masked gains.

## Hypotheses and next steps
1) Decoherence root causes
   - Keys sensitivity: K quant noise perturbs attention logits more than V. Even per-token scales may be too coarse for some layers.
   - Early layers sensitivity: early attention layers amplify quant noise more.
   - Scale dynamics: per-token amax can fluctuate; clipping or percentile-based scales may stabilize.

   Ablations:
   - Keep K in BF16, V in FP8 (timewise). Expect quality to improve notably if K was the main driver.
   - FP8-KV only on late layers (e.g., last 1/3). Check decoherence horizon.
   - Percentile scales (e.g., 99.5th) instead of amax to reduce outlier sensitivity.
   - Per-channel scales for K only.

2) FPS flatline root causes
   - Dequant overhead: per-token scale mul may be memory-bound and not fused; overhead cancels KV savings.
   - Layout conversions: concat/gather before FlexAttention introduces extra memory ops.
   - Non-KV bottlenecks: Q projection, MLPs, VAE decode, or display dominate step time.
   - Compile path: torch.compile retracing or unfused kernels negate benefits.

   Profiling plan:
   - Time breakdown: dequant, attention, MLP, VAE decode, display; report pipeline-only time.
   - Check FlexAttention kernel selection and fusions in logs; fall back to SDPA for A/B.
   - Pre-dequant into persistent contiguous scratch vs on-the-fly; measure difference.
   - Head/tile sizes: ensure `[B,H,T,D]` contiguous; avoid repeated cat/roll in hot loop.

## Action items
- A1: Add K-BF16/V-FP8 variant and late-layers-only toggle; compare decoherence horizon.
- A2: Add percentile scaling option for K; compare.
- A3: Profile per-step time components; verify FlexAttention path and layout costs.

## Conclusion (interim)

## Mitigation knobs we added
- K-only FP8 toggle: `OWL_K_FP8=0` keeps Keys in BF16 while Values use FP8 (timewise). Reduces decoherence risk.
- Late-layers-only FP8: `OWL_KV_LATE_LAYERS=N` enables FP8 only for the last N layers (Keys/Values independently controlled via `OWL_K_FP8`).

### Example runs
```bash
# Baseline profiling (BF16 KV)
OWL_PROFILE_KV=1 OWL_FP8_KV=0 python -m inference.game_cv

# FP8-KV profiling (full), K also FP8
OWL_PROFILE_KV=1 OWL_FP8_KV=1 OWL_K_FP8=1 python -m inference.game_cv

# FP8-KV with K in BF16 (recommended for stability)
OWL_PROFILE_KV=1 OWL_FP8_KV=1 OWL_K_FP8=0 python -m inference.game_cv

# FP8-KV on last 8 layers only, K in BF16
OWL_PROFILE_KV=1 OWL_FP8_KV=1 OWL_K_FP8=0 OWL_KV_LATE_LAYERS=8 python -m inference.game_cv
```

## Results: Phase A baseline (K BF16, V FP8 timewise)

- Command used:
  - `OWL_PROFILE_KV=1 OWL_FP8_KV=1 OWL_K_FP8=0 python -m inference.game_cv`
- Visual quality:
  - Good; decoherence substantially reduced versus full FP8 (K+V). Subjectively stable.
- Throughput:
  - No obvious FPS improvement versus BF16 baseline; decode appears compute-bound and/or allocator retains scratch memory.
- Memory:
  - KV reserved memory reduced relative to BF16 cache; comparable to prior FP8-KV runs (~3.1 GB reserved after warmup). Will archive exact KV-Profile lines in the next run.
- Next steps:
  - Archive profiler lines (`[KV-Profile] ...`) for this setting.
  - Proceed to Phase B: enable K-FP8 on late layers only and evaluate stability.

## Results: Phase B (K FP8 on late layers only)

- Command used:
  - `OWL_PROFILE_KV=1 OWL_FP8_KV=1 OWL_K_FP8=1 OWL_KV_LATE_LAYERS=8 python -m inference.game_cv`
- Visual quality:
  - Worked; acceptable quality per visual inspection. No immediate catastrophic decoherence. Appears comparable to Phase A on short horizons.
- Throughput:
  - No significant FPS change vs Phase A or baseline; still compute-bound.
- Memory:
  - Similar reserved memory to Phase A (expected: K-BF16 in early layers, K-FP8 late). Archive KV-Profile lines in future runs for precise deltas.
- Decision:
  - Phase B is viable. We can either keep N=8 as a safe default or sweep N (e.g., 4/8/12) to find the best quality/footprint trade-off.

## Sweep summary and decision

- Settings tested:
  - V-only FP8 (K BF16): best visual quality, stable. Reserved VRAM ~3.1–3.3 GB after warmup; pipeline FPS unchanged vs baseline.
  - K FP8 on last 4/8/12 layers: visually comparable to V-only FP8; baseline (V-only) still slightly better.
- Decision:
  - Default to V-only FP8 (Keys BF16) across all layers.
  - Keep K-FP8 late-layers as an advanced knob; default `OWL_KV_LATE_LAYERS=0`.
  - Defer stabilizers (percentile/per-channel K scales) since quality is acceptable and not degrading to noise.
  - Focus next on perf profiling to identify true bottlenecks (model compute vs decoder vs layout).

## Optional stabilizers (quality) and flags

Use only if the “stand still” drift is objectionable. Keep K in BF16; apply to V only unless noted.

- Percentile scaling (V only): replace per‑token amax with percentile to reduce outlier sensitivity
  - Flag: `OWL_V_PERCENTILE=99` (or `99.5`)
  - Effect: `scale = percentile(|V|, P) / 127` per head/token

- Late‑layers‑only V FP8 (already supported): reduce early‑layer error accumulation
  - Flag: `OWL_KV_LATE_LAYERS=N` (e.g., `8`), keep `OWL_K_FP8=0`

- Per‑channel scales for V: finer granularity per [H,D] instead of per [H]
  - Flag: `OWL_V_PER_CHANNEL=1`

- Temporal smoothing on scales: avoid sudden scale drops frame‑to‑frame
  - Flag: `OWL_SCALE_SMOOTH=0.9`
  - Effect: `scale_t = max(scale_t, α · scale_{t−1})`

Recommended order:
1) Percentile scaling on V (keep K BF16)
2) Late‑layers‑only V FP8 (tune N)
3) Per‑channel V scales
4) Optional temporal smoothing

## Profiling findings (attention vs decoder vs KV)

- Attention dominates step time after warmup: attn1/attn2 ≈ 5–11 ms per call.
- KV quant/dequant overhead is negligible (q_ms≈0, dq_ms≈0), confirming the KV path is not the bottleneck.
- Reserved memory ~3.2 GB stable post-warmup with FP8-KV; allocator retention explains lack of apparent drop in some runs.
- Next focus: ensure best attention kernels (SDPA/Flash/Flex) are selected; measure decoder cost.

### New findings (latest runs)
- Cold-start compile dominates first 1–2 prints; ignore those.
- Warm loop (V‑only FP8, K BF16):
  - Attention per call ≈ 5–12 ms; two calls per frame.
  - Decoder ≈ 8–9 ms per frame.
  - KV quant/dequant ≈ 0 ms (negligible), reserved VRAM ≈ 3.2 GB.
- Quality: Slight quality delta vs full BF16; when standing still, drift is more visible than baseline (does not collapse to noise).

Implication: We are compute‑bound by transformer attention/MLP and decoder; KV IO is not limiting. FP8‑KV remains a memory/bandwidth win with small quality cost.

### Planned actions (without changing diffusion steps/scheduler)
- Add decoder timing to the profile output (done).
- Verify FlexAttention/SDPA fastpaths consistently; ensure [B,H,T,D] contiguity and avoid extra cat/roll.
- Add optional MLP timing if needed.
- Add SDPA/FlashAttention fastpath on decode (no mask) behind `OWL_ATTN_SDPA_DECODE=1`; keep FlexAttention when block_mask is present.

### Latest profiler snapshot (compile OFF)
- Command:
  - `OWL_COMPILE=0 OWL_PROFILE_KV=1 OWL_PROFILE_KV_FIRST=2 OWL_PROFILE_KV_EVERY=50 OWL_FP8_KV=1 OWL_K_FP8=0 python -m inference.game_cv`
- Notable lines captured:
  - `prefill_s=5.643s, max_reserved=8142.0MB`
  - `step1_s=2.000s (attn1=2182.06ms), step2_s=0.044s (attn2=48.39ms), q_ms=21.09, dq_ms=7.99`
  - `accum: attn_total_ms=6629.54, mlp_total_ms=94.25`
  - `decoder_ms=115.12ms`
  - Warm loop (sample):
    - `step1_s=0.039s (attn1=42.83ms), step2_s=0.041s (attn2=44.80ms), q_ms=24.04, dq_ms=13.05`
    - `accum: attn_total_ms=6680.12, mlp_total_ms=101.77`
    - `decoder_ms=26.58ms`
    - Later: `q_ms=352.48, dq_ms=536.87` (expected higher without compile; confirms compile masks Q/DQ overhead in prior runs)
- Takeaways:
  - Without compile, attention and MLP costs are measurable and dominate; decoder ~25–30ms (uncompiled). With compile ON, decoder ~8–9ms and q/dq ~0ms, matching earlier findings that runtime is compute‑bound by attention+decoder, not KV.

## Roadmap: next steps after stabilization (to finish all phases)

1) Stabilization (now)
- Lock default: V‑only FP8 across all layers; K in BF16.
- If needed: enable stabilizers in order of lowest risk → percentile (V), late‑layers V FP8, per‑channel V, temporal smoothing. Record settings and quality notes in this file.
- Freeze default flags once visually acceptable in “stand still” and normal movement.

2) Phase C: Hardening & configuration
- Make flags first‑class in a config (not only env): `fp8_kv`, `k_fp8`, `kv_late_layers`, `v_percentile`, `v_per_channel`, `scale_smooth_alpha`.
- Add a one‑shot “report settings” line on startup for reproducibility.
- Add a seed + deterministic mode to reproduce stabilization results for QA.

3) Performance focus (no scheduler changes)
- Attention path:
  - Ensure FlexAttention/SDPA fastpath always used; keep tensors `[B,H,T,D]` contiguous. A decode-only SDPA toggle is available via `OWL_ATTN_SDPA_DECODE=1`.
  - Minimize concat/roll before the kernel; reuse buffers.
  - Keep `torch.compile` ON for model; avoid graph breaks from profiling code.
- Decoder path:
  - Time decoder consistently; consider leaving decoder compiled; only optimize if it dominates.
- Optional: TransformerEngine trial to fuse dequant inside attention (retain BF16 activations).

4) Optional Phase D: TensorRT engines (if needed later)
- Only if you need additional perf beyond PyTorch: export prefill/decode graphs, keep activations FP16/BF16, KV in 8‑bit, build engines with FlexAttention.
- Profile vs PyTorch to justify complexity.

Implementation note (current):
- Added optional TensorRT engine for the VAE frame decoder behind `OWL_TRT_DECODER=1`.
  - When enabled, `inference/causvid_pipeline.py` builds a Torch‑TensorRT engine for the decoder on first use (FP16) and falls back to PyTorch if unavailable.
  - We avoid `torch.compile` on the decoder when TRT is enabled to ensure the original module is compiled by TRT.
  - Profiling prints include a build confirmation when `OWL_PROFILE_KV=1`.
  - Model (transformer) remains in PyTorch for now due to KV cache object inputs; future work may stage TRT for fixed‑shape subgraphs.

5) Validation and release checklist
- A/B runs: baseline BF16 vs default FP8 settings; log FPS, decoder_ms, attn_ms, memory, and qualitative notes.
- Save a short “stand still” clip and an active clip for regression checks.
- Set defaults:
  - `OWL_FP8_KV=1`, `OWL_K_FP8=0`, `OWL_KV_LATE_LAYERS=0`.
  - Stabilizers OFF by default; document recommended values (e.g., `OWL_V_PERCENTILE=99.5`) when needed.
- Document rollback: set `OWL_FP8_KV=0` to disable FP8‑KV entirely.

- FP8-KV achieves memory reduction but quality degrades faster vs BF16 and FPS did not improve yet. Focus shifts to K-only FP8 or late-layer FP8, plus profiling to expose non-KV bottlenecks.

- DRAM bytes per decode step scale roughly with `O(n_layers * H * T * D)` for two K/V reads; halving bytes for K and V yields ~2× KV traffic reduction.
- Dequant cost: one fused multiply per element is negligible compared to memory fetch; on consumer GPUs bound by bandwidth, net latency usually improves.

### Regression note (post ring-buffer/tail-read attempt)
- Symptom: output frames went black immediately after the first frame when enabling both ring-buffers and tail-only reads.
- Status: Reverting to legacy roll-based cache restored visuals. Framerate remained good under legacy path.
- Likely cause: ring index misalignment with scales or tail-slice logic (per-token scale alignment) in `QuantizedStaticCache`.

### Decision: remove ring buffer implementation (for now)
- Observation: Ring buffers introduced black frames under `torch.compile` and severe lag spikes due to retraces/dynamic indexing.
- Decision: Remove ring buffer code paths and env flags; revert to legacy roll-based window maintenance. Keep tail-only reads for local layers.
- Rationale: Stability and FPS are better with roll; ring can be revisited via fused/compile-friendly kernels later.

### Debug flags and logging
- `OWL_PROFILE_KV=1` with `OWL_PROFILE_KV_FIRST=3`, `OWL_PROFILE_KV_EVERY=50`.
- `OWL_K_FP8=0` (default; keep Keys in BF16).
- `OWL_KV_LATE_LAYERS=N` to restrict FP8 to last N layers.
- Removed ring toggles; only legacy roll is supported now.
- Log lines to capture on each update/get:
  - `update_begin/update_done`: `layer`, `new_len`, `wp`, `filled`, `offset`.
  - `get`: `layer`, `logical_len`, shapes; `get_tail_begin`: `tail_len`, `filled`, `wp`.
  - Scale NaN fraction: `nan_frac` in debug prints.

### Rollback
- If any step regresses visuals, disable that step via the above flags and proceed with the next safest variant (e.g., BF16 ring only).


