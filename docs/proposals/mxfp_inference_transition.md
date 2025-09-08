Title: MXFP8 → MXFP4 Inference Transition Plan (TorchAO) for GameRFT (DiT)

Scope
- Target entrypoint: `inference/game_cv.py` → `CausvidPipeline`.
- Model: `owl_wms.from_pretrained(...)` → `GameRFT` with `GameRFTCore` (DiT backbone) and VAE frame decoder.
- Targeted ops: Transformer `nn.Linear` GEMMs (QKV projections, output projections, MLPs). Attention kernel (FlexAttention/SDPA) remains BF16 compute.
- Coexistence: This plan is orthogonal to FP8 KV-cache work (`docs/proposals/fp8_kv_flexattention.md`). We can enable both: MXFP8 for GEMMs + 8‑bit KV storage, while keeping activations BF16.

Why MX formats (MXFP8 → MXFP4)
- MXFP8: Per‑tile scaled FP8 math via CUTLASS/TRITON kernels; lowers bandwidth and can improve GEMM throughput on modern NVIDIA GPUs while keeping conversion overhead low.
- MXFP4: Higher compression and potential perf on specific kernels, but higher numerical risk. Plan a staged rollout after MXFP8 stabilizes.

Pre‑requisites
- GPU: NVIDIA Ampere or newer (SM80+ recommended for CUTLASS FP8 kernels; Hopper SM90 preferred).
- PyTorch: Recent nightly/stable supporting TorchAO prototype. Verify `torchao.prototype.mx_formats` is present.
- Install TorchAO (prototype) and CUTLASS dependencies if needed.
- Keep `torch.compile` interaction in mind: apply MX transforms before compiling the model, and avoid dynamic code in hot paths.

High‑level plan
1) Baseline capture (BF16)
   - Lock baseline FPS/latency, memory, and quality (PSNR/SSIM, drift notes). Save environment info, commit SHA, GPU, driver, torch/torchao versions.

2) MXFP8 (MLP‑only)
   - Transform only MLP `nn.Linear` layers to MXFP8. Keep QKV/out_proj BF16 initially.
   - Validate numerics and perf; ensure no regressions in visuals.

3) MXFP8 (MLP + QKV/out_proj)
   - Transform all transformer `nn.Linear` layers, including QKV and output projections.
   - Re‑profile and A/B test vs step 2.

4) MXFP8 + FP8‑KV (optional combo)
   - Combine with the quantized KV cache from `fp8_kv_flexattention.md`. Expect memory savings; perf benefit depends on attention kernel.

5) MXFP4 (MLP‑only, opt‑in)
   - Trial MXFP4 on MLP layers only. Gate behind a strict flag. Validate quality (riskier numerics).

6) MXFP4 (broader)
   - Expand cautiously (late layers only or MLP+out_proj), based on results.

Implementation sketch
- Apply transforms immediately after model creation/load and before `torch.compile` or TensorRT hooks.
- Suggested flags (env or YAML config under `inference`):
  - `OWL_MXFP_ENABLE=0/1` (default 0)
  - `OWL_MXFP_BITS=8|4` (default 8)
  - `OWL_MXFP_SCOPE=mlp|attn|all` (default `mlp`)
  - `OWL_MXFP_KERNEL=cutlass|triton` (default `cutlass`)
  - `OWL_MXFP_LIST=1` (log transformed modules)

Config wiring
- `owl_wms/configs.py` now exposes `InferenceConfig` fields for MX and FP8‑KV with defaults matching our best run so far:
  - `mxfp_enable: true`, `mxfp_bits: 8`, `mxfp_scope: "all"`, `mxfp_kernel: "cutlass"`
  - `fp8_kv: true`, `k_fp8: false`, `kv_late_layers: 12`
- `inference/causvid_pipeline.py` adopts these defaults at startup if env vars are unset, then applies MX transforms before compile.

Minimal code example (MXFP8, CUTLASS)
```python
import torch
import torch.nn as nn
from torchao.prototype.mx_formats.inference_workflow import (
    MXFPInferenceConfig,
    _mx_inference_linear_transform,
    MXGemmKernelChoice,
)

def apply_mx_transforms(model: nn.Module, scope: str = "mlp", bits: int = 8, kernel: str = "cutlass") -> int:
    assert bits in (4, 8)
    kernel_choice = MXGemmKernelChoice.CUTLASS if kernel == "cutlass" else MXGemmKernelChoice.TRITON
    cfg = MXFPInferenceConfig(gemm_kernel_choice=kernel_choice, bits=bits)

    def should_transform(name: str, mod: nn.Module) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False
        name_l = name.lower()
        if scope == "mlp":
            # Heuristic: include MLP linears, exclude qkv/out_proj
            return ("mlp" in name_l) or ("fc" in name_l and "qkv" not in name_l)
        if scope == "attn":
            return ("qkv" in name_l) or ("to_q" in name_l) or ("to_k" in name_l) or ("to_v" in name_l) or ("out_proj" in name_l)
        return True  # all

    transformed = 0
    for name, mod in model.named_modules():
        if should_transform(name, mod):
            _mx_inference_linear_transform(mod, cfg)
            transformed += 1
    return transformed
```

Where to call
- `inference/game_cv.py` or the pipeline assembly path that builds/loads the model:
  1) Load model.
  2) If `OWL_MXFP_ENABLE=1`: call `apply_mx_transforms(model, ...)`.
  3) Optionally print transformed module count and sample names if `OWL_MXFP_LIST=1`.
  4) Compile the model if `inference.compile=true`.

Behavioral notes
- Input/output dtypes remain BF16 at module boundaries; kernels handle low‑precision math internally.
- Keep attention kernel unchanged (FlexAttention/SDPA). QKV/out_proj linears may be transformed in phase 3.
- Ensure tensors feeding the attention kernels remain `[B,H,T,D]` contiguous, as in the baseline.

Testing and validation
1) Unit (module level)
   - Reference vs MX output for representative `nn.Linear` shapes. Metrics: mean absolute error (MAE), relative error, max error. Run on random BF16 inputs with seed.
   - Saturation/scale stats if available from TorchAO (or add hooks to count clamped values).

2) Microbench (perf)
   - Time forward passes for typical shapes (MLP inner: `[B*T, D] @ [D, 4D]`, QKV: `[B*T, D] @ [D, 3H*Dh]`).
   - Report throughput vs BF16 over N iters, warmup K.

3) Integration (pipeline)
   - A/B vs BF16: FPS, attention/MLP/decoder ms (there are existing profiling prints for attention/decoder; extend to MLP if needed).
   - Memory: `torch.cuda.max_memory_reserved()` after warmup.
   - Quality: PSNR/SSIM on short rollouts; qualitative drift notes (“stand still” and motion).

4) Stability
   - `torch.compile` ON/OFF runs to ensure transforms don’t induce graph breaks or retracing. If issues, apply transforms before compile and avoid dynamic control flow around transformed modules.

Logging we should capture every run
- System: GPU model, driver, CUDA, torch, torchao, cutlass/triton backend.
- Repo: commit SHA, config YAML path, relevant env flags.
- Transform: number of linears transformed, scope, bits, kernel, sample module names.
- Perf: FPS, attention_ms, mlp_ms, decoder_ms, total step time; microbench results (where applicable).
- Memory: max_reserved, allocated deltas.
- Numerics: error stats on sampled layers (optional), saturation counts if accessible.
- Incidents: errors encountered, fallbacks taken.

Example flags / runs
```bash
# Baseline BF16
OWL_PROFILE_KV=1 OWL_MXFP_ENABLE=0 python -m inference.game_cv

# MXFP8 on MLP only (recommended first step)
OWL_PROFILE_KV=1 OWL_MXFP_ENABLE=1 OWL_MXFP_BITS=8 OWL_MXFP_SCOPE=mlp OWL_MXFP_KERNEL=cutlass python -m inference.game_cv

# MXFP8 on all linears (QKV+MLP+out_proj)
OWL_PROFILE_KV=1 OWL_MXFP_ENABLE=1 OWL_MXFP_BITS=8 OWL_MXFP_SCOPE=all python -m inference.game_cv

# MXFP4 on MLP only (experimental)
OWL_PROFILE_KV=1 OWL_MXFP_ENABLE=1 OWL_MXFP_BITS=4 OWL_MXFP_SCOPE=mlp python -m inference.game_cv
```

Rollout gates (go/no‑go)
- Gate A (MXFP8 MLP‑only): Proceed if FPS ≥ baseline − 3% and quality within tolerance (no visible artifacts; PSNR drop ≤ 0.3 dB).
- Gate B (MXFP8 all linears): Proceed if additional PSNR drop ≤ 0.3 dB and FPS ≥ Gate A.
- Gate C (MXFP8 + FP8‑KV): Proceed if quality acceptable and memory target improved; FPS change may be neutral if attention dominates.
- Gate D (MXFP4 MLP‑only): Proceed only if PSNR drop ≤ 0.5 dB and no instabilities on 3+ test clips.

Backtracking guide
- If quality regresses:
  - Restrict scope to MLP‑only, or exclude early layers.
  - Keep QKV/out_proj in BF16; try MLP only.
  - Prefer MXFP8 over MXFP4; avoid MXFP4 on attention projections initially.
- If perf regresses:
  - Confirm CUTLASS kernels are used; try TRITON alternative if available.
  - Ensure input tensors are contiguous and avoid extra casts around transformed modules.
  - Re‑test with `torch.compile` ON (and OFF) to isolate compile interactions.

File‑by‑file proposal (edits later)
1) `inference/causvid_pipeline.py`
   - Read flags, call `apply_mx_transforms(model, scope, bits, kernel)` after model load and before compile.
   - Print a short summary if enabled.

2) `owl_wms/configs.py` (or inference config plumbing)
   - Add config fields under `inference`: `mxfp_enable`, `mxfp_bits`, `mxfp_scope`, `mxfp_kernel` and env overrides.

3) Optional profiling helpers
   - Add a small timing wrapper around MLP forward to report `mlp_ms` next to existing attention/decoder timings.

MXFP8/MXFP4 for KV cache (experiments)
- Objective: Evaluate storing K/V in MX formats (FP8 and FP4 variants) instead of BF16 or int8+scale, while keeping attention math in BF16. Dequantize K/V to BF16 just‑in‑time before FlexAttention/SDPA.
- Motivation vs current FP8‑KV approach: MX formats may offer better HW kernel alignment and simpler packing than custom int8+scale. Risk is higher quantization error, especially for Keys.

Design options
1) MXFP8 storage with per‑token scales (recommended start)
   - Format: V → E4M3; K → E5M2. Per‑token, per‑head scales (amax over D) to stabilize early steps and ensure alignment across time.
   - Update path: Quantize newly produced BF16 `k,v` into MXFP8 tiles; record scales alongside tokens (time axis). Maintain window via roll (no ring buffer initially).
   - Get path: Dequantize the active window into a contiguous BF16 scratch buffer `[B,H,T,D]` for attention.

2) MXFP8 storage with per‑head EMA scales
   - Cheaper metadata but more drift risk. Consider only after per‑token variant is stable.

3) MXFP4 storage (experimental)
   - Start with V‑only FP4 (per‑token scales), keep K in BF16. If stable, trial K FP4 on late layers only.

4) Mixed formats
   - K in BF16, V in MXFP8/MXFP4. Or K in MXFP8 (late layers only), V in MX.

API sketch (augment cache)
```python
# Pseudocode extensions to QuantizedStaticCache
class QuantizedStaticCache:
    def __init__(self, ..., kv_storage: str = "bf16", kv_bits: int = 8, k_fmt: str = "e5m2", v_fmt: str = "e4m3", per_token: bool = True):
        # kv_storage: "bf16" | "i8_scale" | "mxfp"
        # when kv_storage == "mxfp", use kv_bits in {8,4} and formats per tensor
        ...

    @torch.no_grad()
    def update(self, new_k, new_v, layer_ind):
        if self.kv_storage == "mxfp":
            # compute scales (per-token or per-head) and quantize to mxfp{bits}
            qk, sk = quantize_mx(new_k, bits=self.kv_bits, fmt=self.k_fmt, per_token=self.per_token)
            qv, sv = quantize_mx(new_v, bits=self.kv_bits, fmt=self.v_fmt, per_token=self.per_token)
            self._roll_write(layer_ind, qk, qv, sk, sv)
        elif self.kv_storage == "i8_scale":
            ... # existing int8+scale path
        else:
            ... # bf16

    @torch.no_grad()
    def get(self, layer_ind, new_k=None, new_v=None):
        if self.kv_storage == "mxfp":
            k = dequantize_mx(self.k_mx[layer_ind], self.scale_k[layer_ind])
            v = dequantize_mx(self.v_mx[layer_ind], self.scale_v[layer_ind])
            if new_k is not None:
                k = torch.cat([k, new_k], dim=2)
                v = torch.cat([v, new_v], dim=2)
            return k, v
        ...
```

Experiment matrix (KV)
- K BF16, V MXFP8 per‑token (all layers) → default starting point.
- K/V MXFP8 per‑token on last N layers only (N ∈ {4, 8, 12}).
- K/V MXFP8 per‑head EMA (if quality comparable to per‑token).
- V MXFP4 per‑token, K BF16 (experimental).
- K MXFP8 (late layers), V MXFP8 per‑token.

Metrics and logging (KV)
- Quant stats: per‑layer saturation counts (|q|=max), scale min/mean/max over time, NaN/inf guards.
- Timing: `kv_update_ms`, `kv_get_ms`, dequant bytes processed per step; ensure prints align with existing KV profiler.
- Memory: reserved bytes after warmup; estimated KV window bytes by format.
- Quality: PSNR/SSIM vs BF16 KV baseline; subjective drift notes.

Gates (KV)
- Proceed beyond V‑only MXFP8 if PSNR drop ≤ 0.3 dB and no visible artifacts.
- Only attempt MXFP4 on V after MXFP8 is acceptable; keep K in BF16 unless late‑layers K MXFP8 proves stable.

Interplay with GEMM MX
- Test KV MX independently first (model GEMMs BF16). Then combine with MXFP8 linears. Confirm no regressions due to increased quantization compounding.

Fallbacks
- At any sign of instability: revert K to BF16; reduce N layers; switch per‑head EMA back to per‑token scales; disable MXFP4.

MXFP4 transition notes
- Start with MLP linears only; avoid attention projections initially.
- Expect higher error; consider per‑layer allowlist (late layers first) and quick rollback flag.
- Use stricter gates; keep BF16 fallback ready.

Risks
- Kernel availability differs by arch/driver; CUTLASS/TRITON selection matters.
- Dynamic shapes and `torch.compile` graph breaks can undermine perf.
- MXFP4 may introduce visible artifacts; treat as experimental until proven stable.

Incident log template (to append per run)
- Hardware/Software: [GPU, driver, CUDA, torch, torchao]
- Config: [flags, YAML, scope, bits, kernel]
- Transform summary: [count, names sample]
- Metrics: [FPS, attn_ms, mlp_ms, decoder_ms, memory]
- Quality: [PSNR/SSIM, drift notes]
- Issues: [errors, fallbacks, kernel changes]

Appendix: TorchAO install and references
- Installation: Prefer the PyTorch nightly + TorchAO prototype matching your CUDA. If needed, clone TorchAO and install editable. Keep this as a last resort; upstream wheels are preferred.
- API example we rely on:
```python
from torchao.prototype.mx_formats.inference_workflow import MXFPInferenceConfig, _mx_inference_linear_transform, MXGemmKernelChoice

cfg = MXFPInferenceConfig(gemm_kernel_choice=MXGemmKernelChoice.CUTLASS)
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear) and mod.weight.dtype == torch.bfloat16:
        _mx_inference_linear_transform(mod, cfg)
```

What success looks like
- MXFP8 MLP‑only: neutral or improved FPS, no visible quality loss, identical UX.
- MXFP8 all linears: modest FPS uptick over MLP‑only, quality acceptable.
- MXFP4 MLP‑only: measurable perf/memory benefit on supported kernels with tolerable quality drop on short rollouts.

Results: MXFP8 all linears (fresh compile)
- Command used:
```bash
OWL_MXFP_ENABLE=1 OWL_MXFP_BITS=8 OWL_MXFP_SCOPE=all OWL_MXFP_LIST=1 python -m inference.game_cv
```
- Observations (sample lines):
  - Cold compile/autotune latency lines around ~28–29 s.
  - Steady‑state examples:
    - `FPS (pipeline): 70–75`, latency ~11–12 ms
    - `FPS (pipeline): 55–66`, latency ~18–26 ms (scene‑dependent)
- Note: Gains likely confounded by recompile; re‑A/B with compile held constant.

Results: MXFP4 MLP‑only (fresh compile)
- Command used:
```bash
OWL_MXFP_ENABLE=1 OWL_MXFP_BITS=4 OWL_MXFP_SCOPE=mlp OWL_MXFP_LIST=1 python -m inference.game_cv
```
- Note: Fresh compile; treat steady‑state only for comparisons. Re‑run with identical compile settings for a proper A/B.

Next phase: MXFP8 all linears + FP8‑KV (V‑only)
- Goal: Combine GEMM MXFP8 with V‑only FP8 KV (K in BF16) to check end‑to‑end perf/memory.
- Command:
```bash
OWL_COMPILE=1 OWL_PROFILE_KV=1 OWL_MXFP_ENABLE=1 OWL_MXFP_BITS=8 OWL_MXFP_SCOPE=all \
OWL_FP8_KV=1 OWL_K_FP8=0 OWL_KV_LATE_LAYERS=12 python -m inference.game_cv
```


