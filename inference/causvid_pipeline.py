from owl_wms.configs import Config
from owl_wms.data import get_loader
from owl_wms import from_pretrained
from owl_wms.nn.kv_cache import KVCache, StaticCache, QuantizedStaticCache
from owl_wms.nn.rope import cast_rope_buffers_to_fp32
from owl_wms.nn.mxfp import apply_mx_transforms
from owl_wms.nn.attn import get_block_mask

import torch.nn.functional as F
import torch

import random
import torch
from accelerate import init_empty_weights
import os
import time
from copy import deepcopy
import glob
import gc

def zlerp(x, alpha):
    return x * (1. - alpha) + alpha * torch.randn_like(x)

@torch.no_grad()
def to_bgr_uint8(frame, target_size=(1080,1920)):
    # frame is [rgb,h,w] in [-1,1]
    frame = frame.flip(0)
    frame = frame.permute(1,2,0)
    frame = (frame + 1) * 127.5
    frame = frame.clamp(0, 255).to(device='cpu',dtype=torch.uint32,memory_format=torch.contiguous_format,non_blocking=True)
    return frame

SAMPLING_STEPS = 2
WINDOW_SIZE = 60

class CausvidPipeline:
    def __init__(self, cfg_path="configs/dit_v4_dmd.yml", ckpt_path="vid_dit_v4_dmd_7k.pt", ground_truth = False):
        cfg = Config.from_yaml(cfg_path)
        model_cfg = cfg.model
        train_cfg = cfg.train
        infer_cfg = cfg.inference if hasattr(cfg, "inference") and cfg.inference is not None else None

        self.ground_truth = ground_truth
        
        self.model, self.frame_decoder = from_pretrained(cfg_path, ckpt_path, True)
        self.model = self.model.core.cuda().bfloat16().eval()
        cast_rope_buffers_to_fp32(self.model)

        self.frame_decoder = self.frame_decoder.cuda().bfloat16().eval()

        # Store scales as instance variables
        self.frame_scale = train_cfg.vae_scale
        self.image_scale = train_cfg.vae_scale
        self.mouse_scaler = 1.0

        self.history_buffer = None
        self.mouse_buffer = None
        self.button_buffer = None

        # Ground truth specific buffers
        if self.ground_truth:
            self.future_mouse_buffer = None
            self.future_button_buffer = None
            self.gt_step = 0

        self.alpha = 0.25

        # Adopt inference config defaults into environment if not already set
        if infer_cfg is not None:
            def _set_default_env(key: str, value):
                if os.environ.get(key) is None and value is not None:
                    os.environ[key] = str(value)
            _set_default_env("OWL_COMPILE", 1 if infer_cfg.compile else 0)
            _set_default_env("OWL_PROFILE_KV", 1 if infer_cfg.profile_kv else 0)
            _set_default_env("OWL_PROFILE_KV_EVERY", infer_cfg.profile_kv_every)
            _set_default_env("OWL_PROFILE_KV_FIRST", infer_cfg.profile_kv_first)
            _set_default_env("OWL_SAMPLING_STEPS", infer_cfg.sampling_steps)
            # FP8 KV
            _set_default_env("OWL_FP8_KV", 1 if infer_cfg.fp8_kv else 0)
            _set_default_env("OWL_K_FP8", 1 if infer_cfg.k_fp8 else 0)
            _set_default_env("OWL_KV_LATE_LAYERS", infer_cfg.kv_late_layers)
            _set_default_env("OWL_KV_STORAGE", infer_cfg.kv_storage)
            _set_default_env("OWL_KV_BITS", infer_cfg.kv_bits)
            _set_default_env("OWL_KV_TAIL_ONLY", 1 if getattr(infer_cfg, 'kv_tail_only', True) else 0)
            # TRT
            _set_default_env("OWL_TRT_DECODER", 1 if infer_cfg.trt_decoder else 0)
            _set_default_env("OWL_TRT_DECODER_FORCE", 1 if infer_cfg.trt_decoder_force else 0)
            _set_default_env("OWL_TRT_DECODER_SLOW_PCT", infer_cfg.trt_decoder_slow_pct)
            # MXFP
            _set_default_env("OWL_MXFP_ENABLE", 1 if infer_cfg.mxfp_enable else 0)
            _set_default_env("OWL_MXFP_BITS", infer_cfg.mxfp_bits)
            _set_default_env("OWL_MXFP_SCOPE", infer_cfg.mxfp_scope)
            _set_default_env("OWL_MXFP_KERNEL", infer_cfg.mxfp_kernel)
            _set_default_env("OWL_MXFP_LATE_LAYERS", infer_cfg.mxfp_late_layers)
            # Attention fastpath
            _set_default_env("OWL_ATTN_SDPA_DECODE", 1 if getattr(infer_cfg, 'attn_sdpa_decode', False) else 0)

        # Optional MXFP8 transforms (apply before compile)
        try:
            transformed = apply_mx_transforms(self.model)
            if transformed > 0:
                print(f"[MXFP] Applied transforms to {transformed} Linear modules (scope={os.environ.get('OWL_MXFP_SCOPE', 'mlp')})")
        except Exception as e:
            # Non-fatal; proceed without MX transforms
            print(f"[MXFP] Skipping MX transforms due to error: {e}")

        # Optional compile toggle
        self.compile_enabled = bool(int(os.environ.get("OWL_COMPILE", "1")))
        if self.compile_enabled:
            self.model = torch.compile(self.model)#, mode = 'max-autotune', dynamic = False, fullgraph = True)
            self.frame_decoder = torch.compile(self.frame_decoder, mode = 'max-autotune', dynamic = False, fullgraph = True)
        
        self.device = 'cuda'
        
        self.cache = None
        self.profile_kv = bool(int(os.environ.get("OWL_PROFILE_KV", "0")))
        self.profile_kv_every = int(os.environ.get("OWL_PROFILE_KV_EVERY", "30"))
        self.profile_kv_first = int(os.environ.get("OWL_PROFILE_KV_FIRST", "3"))
        self._profile_kv_count = 0
        self.use_fp8_kv = bool(int(os.environ.get("OWL_FP8_KV", "0")))
        # Optional TensorRT acceleration flags for decoder
        self.use_trt_decoder = bool(int(os.environ.get("OWL_TRT_DECODER", "0")))
        self.trt_force = bool(int(os.environ.get("OWL_TRT_DECODER_FORCE", "0")))
        self.trt_slow_pct = float(os.environ.get("OWL_TRT_DECODER_SLOW_PCT", "10"))
        self._trt_decoder_built = False
        self._trt_decoder = None
        self._trt_benchmark = None  # {'pytorch_ms':..., 'trt_ms':...}
        # One-time backend sanity
        if self.profile_kv:
            try:
                sk = getattr(torch.backends.cuda, "sdp_kernel", None)
                if sk is not None:
                    print("[Kernels] sdp_kernel namespace present; SDPA backends configurable")
                else:
                    print("[Kernels] sdp_kernel namespace not present; using default SDPA backends")
            except Exception:
                print("[Kernels] backend query unavailable; proceeding without kernel print")
        self._initial_history_buffer = None
        self._initial_mouse_buffer = None
        self._initial_button_buffer = None
        if self.ground_truth:
            self._initial_future_mouse_buffer = None
            self._initial_future_button_buffer = None

        self.init_buffers()

        self.prev_frame = None
        self.prev_mouse = None
        self.prev_btn = None
    
    def _build_cache(self):
        # Build cache similar to av_caching_v2.py
        self.model.transformer.disable_decoding()
        batch_size = 1
        init_len = self.history_buffer.size(1)
        
        # Initialize KV cache
        if self.use_fp8_kv:
            # max_length is number of frames (same as StaticCache usage)
            # Default: K BF16 (k_fp8=0), V FP8 on last 12 layers
            kv_late_layers = int(os.environ.get("OWL_KV_LATE_LAYERS", "12"))
            k_fp8 = bool(int(os.environ.get("OWL_K_FP8", "0")))
            self.cache = QuantizedStaticCache(
                self.model.config,
                max_length = init_len,
                batch_size = batch_size,
                kv_late_layers = kv_late_layers,
                k_fp8 = k_fp8,
            )
        else:
            self.cache = StaticCache(self.model.config, max_length = init_len, batch_size = batch_size)
        #self.cache = KVCache(self.model.config)
        #self.cache.reset(batch_size)
        
        # Noise the history buffer for caching
        prev_x_noisy = zlerp(self.history_buffer, self.alpha)
        prev_t = self.history_buffer.new_full((batch_size, init_len), self.alpha)
        
        # Cache the context
        self.cache.enable_cache_updates()
        if self.profile_kv:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
        _ = self.model(
            prev_x_noisy,
            prev_t,
            self.mouse_buffer,
            self.button_buffer,
            kv_cache=self.cache
        )
        if self.profile_kv:
            torch.cuda.synchronize()
            prefill_s = time.time() - t0
            mem_mb = torch.cuda.max_memory_reserved() / (1024**2)
            print(f"[KV-Profile] prefill_s={prefill_s:.3f}s, max_reserved={mem_mb:.1f}MB, fp8_kv={self.use_fp8_kv}")
        self.cache.disable_cache_updates()
        self.model.transformer.enable_decoding()

        # Legacy path: no special tail-only wiring

        # Do garbage collection
        gc.collect()
        torch.cuda.empty_cache()

    def init_buffers(self, window_size=WINDOW_SIZE):        
        cache_files = glob.glob("data_cache/*.pt")
        if not cache_files:
            raise RuntimeError("No cache files found in data_cache/")
            
        # Randomly select one cache file
        cache_path = random.choice(cache_files)
        
        # Load cached tensors with memory mapping
        cache = torch.load(cache_path, map_location='cpu', mmap=True)
        
        # Extract tensors: vid [n,c,h,w], mouse [n,2], btn [n,11]
        vid = cache["vid"]
        mouse = cache["mouse"]
        button = cache["button"]
        
        # Get a random window from the sample
        seq_len = vid.size(0)
        
        if self.ground_truth:
            # For ground truth, we need extra data for future controls
            future_size = 1000  # Number of future steps to load
            required_len = window_size + future_size
            if seq_len < required_len:
                raise ValueError(f"Sample {cache_path} has length {seq_len} < required_len {required_len}")
            
            start_idx = random.randint(0, seq_len - required_len)
            history_end_idx = start_idx + window_size
            future_end_idx = start_idx + required_len
            
            # Extract history windows and add batch dimension
            self.history_buffer = vid[start_idx:history_end_idx].unsqueeze(0)  # [1,window_size,c,h,w]
            self.mouse_buffer = mouse[start_idx:history_end_idx].unsqueeze(0)  # [1,window_size,2]
            self.button_buffer = button[start_idx:history_end_idx].unsqueeze(0)  # [1,window_size,11]
            
            # Extract future controls
            self.future_mouse_buffer = mouse[history_end_idx:future_end_idx].unsqueeze(0)  # [1,future_size,2]
            self.future_button_buffer = button[history_end_idx:future_end_idx].unsqueeze(0)  # [1,future_size,11]
            
            # Initialize ground truth step counter
            self.gt_step = 0
        else:
            if seq_len < window_size:
                raise ValueError(f"Sample {cache_path} has length {seq_len} < window_size {window_size}")
            
            start_idx = random.randint(0, seq_len - window_size)
            end_idx = start_idx + window_size
            
            # Extract matching windows and add batch dimension
            self.history_buffer = vid[start_idx:end_idx].unsqueeze(0)  # [1,window_size,c,h,w]
            self.mouse_buffer = mouse[start_idx:end_idx].unsqueeze(0)  # [1,window_size,2]
            self.button_buffer = button[start_idx:end_idx].unsqueeze(0)  # [1,window_size,11]

        # Scale buffers (ensure they're on cuda and in bfloat16)
        self.history_buffer = self.history_buffer.cuda().bfloat16() / self.frame_scale
        self.mouse_buffer = self.mouse_buffer.cuda().bfloat16()
        self.button_buffer = self.button_buffer.cuda().bfloat16()
        
        if self.ground_truth:
            self.future_mouse_buffer = self.future_mouse_buffer.cuda().bfloat16()
            self.future_button_buffer = self.future_button_buffer.cuda().bfloat16()

        self._initial_history_buffer = self.history_buffer.clone()
        self._initial_mouse_buffer = self.mouse_buffer.clone()
        self._initial_button_buffer = self.button_buffer.clone()
        if self.ground_truth:
            self._initial_future_mouse_buffer = self.future_mouse_buffer.clone()
            self._initial_future_button_buffer = self.future_button_buffer.clone()

        self._build_cache()

    def restart_from_buffer(self):
        """Restore buffers to their initial state."""
        self.history_buffer = self._initial_history_buffer.clone()
        self.mouse_buffer = self._initial_mouse_buffer.clone()
        self.button_buffer = self._initial_button_buffer.clone()
        
        if self.ground_truth:
            self.future_mouse_buffer = self._initial_future_mouse_buffer.clone()
            self.future_button_buffer = self._initial_future_button_buffer.clone()
            self.gt_step = 0

        self._build_cache()


    @torch.no_grad()
    def __call__(self, new_mouse, new_btn):
        """
        new_mouse is [2,] bfloat16 tensor (assume cuda for both)
        new_btn is [11,] bool tensor indexing into [W,A,S,D,LSHIFT,SPACE,R,F,E,LMB,RMB] (i.e. true if key is currently pressed, false otherwise)

        return frame as [c,h,w] tensor in [-1,1]
        """
        if self.ground_truth:
            # Use ground truth controls instead of player inputs
            if self.gt_step >= self.future_mouse_buffer.size(1):
                raise ValueError("Ground truth data exhausted")
            
            gt_mouse = self.future_mouse_buffer[0, self.gt_step]  # [2]
            gt_btn = self.future_button_buffer[0, self.gt_step]   # [11]
            
            new_mouse_input = gt_mouse[None,None,:]  # [1,1,2]
            new_btn_input = gt_btn[None,None,:]      # [1,1,11]
            
            self.gt_step += 1
        else:
            # Use player inputs
            new_mouse = new_mouse.bfloat16()
            new_btn = new_btn.bfloat16() * self.mouse_scaler

            # Prepare new frame inputs
            new_mouse_input = new_mouse[None,None,:]  # [1,1,2]
            new_btn_input = new_btn[None,None,:]      # [1,1,11]

        # Initialize new frame as noise
        curr_x = torch.randn_like(self.history_buffer[:,:1])  # [1,1,c,h,w]
        curr_t = torch.ones(1, 1, device=curr_x.device, dtype=curr_x.dtype)  # [1,1]

        dt = 1.0 / SAMPLING_STEPS

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        # First sampling step
        if self.profile_kv:
            torch.cuda.reset_peak_memory_stats()
            t_kv0 = time.time()
            e_attn0 = torch.cuda.Event(enable_timing=True); e_attn1 = torch.cuda.Event(enable_timing=True)
            e_attn0.record()
        pred_v = self.model(
            curr_x,
            curr_t,
            new_mouse_input,
            new_btn_input,
            kv_cache=self.cache
        )
        if self.profile_kv:
            e_attn1.record(); torch.cuda.synchronize()
            step1_s = time.time() - t_kv0
            attn1_ms = e_attn0.elapsed_time(e_attn1)
            mem1_mb = torch.cuda.max_memory_reserved() / (1024**2)
        
        curr_x = curr_x - 0.75 * pred_v
        curr_t = curr_t - 0.75

        # Second sampling step does cache update as well
        self.cache.enable_cache_updates()
        if self.profile_kv:
            torch.cuda.reset_peak_memory_stats()
            t_kv1 = time.time()
            e_attn2 = torch.cuda.Event(enable_timing=True); e_attn3 = torch.cuda.Event(enable_timing=True)
            e_attn2.record()
        pred_v = self.model(
            curr_x,
            curr_t,
            new_mouse_input,
            new_btn_input,
            kv_cache=self.cache
        )
        self.cache.disable_cache_updates()
        if self.profile_kv:
            e_attn3.record(); torch.cuda.synchronize()
            step2_s = time.time() - t_kv1
            attn2_ms = e_attn2.elapsed_time(e_attn3)
            mem2_mb = torch.cuda.max_memory_reserved() / (1024**2)
            # Throttle printing: first N frames, then every M frames
            self._profile_kv_count += 1
            should_print = (self._profile_kv_count <= self.profile_kv_first) or (self._profile_kv_count % self.profile_kv_every == 0)
            if should_print:
                # Cache-level quant/dequant accumulated timings
                q_ms = getattr(self.cache, 'quant_ms', 0.0)
                dq_ms = getattr(self.cache, 'dequant_ms', 0.0)
                attn_total = getattr(self.cache, 'attn_ms', 0.0)
                mlp_total = getattr(self.cache, 'mlp_ms', 0.0)
                print(f"[KV-Profile] step1_s={step1_s:.3f}s (attn1={attn1_ms:.2f}ms), step2_s={step2_s:.3f}s (attn2={attn2_ms:.2f}ms), q_ms={q_ms:.2f}, dq_ms={dq_ms:.2f}, max_reserved={max(mem1_mb, mem2_mb):.1f}MB")
                print(f"[KV-Profile] accum: attn_total_ms={attn_total:.2f}, mlp_total_ms={mlp_total:.2f}")

        new_frame = curr_x - 0.25 * pred_v

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds

        # Decode frame for display (profile if enabled)
        if self.profile_kv:
            e_dec0 = torch.cuda.Event(enable_timing=True); e_dec1 = torch.cuda.Event(enable_timing=True)
            e_dec0.record()
        x_to_dec = new_frame[0] * self.image_scale
        # Build TensorRT decoder on first use if enabled
        if self.use_trt_decoder and not self._trt_decoder_built:
            try:
                import torch_tensorrt.dynamo as torchtrt
                example = x_to_dec.contiguous().cuda().half()
                self._trt_decoder = torchtrt.compile(
                    self.frame_decoder.half().eval(),
                    inputs=[example],
                    enabled_precisions={torch.float16},
                    workspace_size=1 << 28,
                )
                self._trt_decoder_built = True
                if self.profile_kv:
                    print("[TensorRT] Decoder engine built (FP16)")
            except Exception as e:
                if self.profile_kv:
                    print(f"[TensorRT] Decoder build failed, falling back to PyTorch: {e}")
                self._trt_decoder = None
                self._trt_decoder_built = True

        if self._trt_decoder is not None:
            if self._trt_benchmark is None and not self.trt_force:
                try:
                    x_pt = x_to_dec.contiguous()
                    x_trt = x_to_dec.contiguous().half()
                    e_p0 = torch.cuda.Event(enable_timing=True); e_p1 = torch.cuda.Event(enable_timing=True)
                    e_t0 = torch.cuda.Event(enable_timing=True); e_t1 = torch.cuda.Event(enable_timing=True)
                    e_p0.record(); _ = self.frame_decoder(x_pt); e_p1.record(); torch.cuda.synchronize(); pt_ms = e_p0.elapsed_time(e_p1)
                    e_t0.record(); _ = self._trt_decoder(x_trt); e_t1.record(); torch.cuda.synchronize(); trt_ms = e_t0.elapsed_time(e_t1)
                    self._trt_benchmark = {"pytorch_ms": pt_ms, "trt_ms": trt_ms}
                    if self.profile_kv:
                        print(f"[TensorRT] bench: pytorch_ms={pt_ms:.2f}, trt_ms={trt_ms:.2f}")
                    if trt_ms > pt_ms * (1.0 + self.trt_slow_pct / 100.0):
                        if self.profile_kv:
                            print("[TensorRT] Slower than PyTorch, disabling TRT decoder.")
                        self._trt_decoder = None
                except Exception as e:
                    if self.profile_kv:
                        print(f"[TensorRT] Benchmark failed: {e}")
            if self._trt_decoder is not None:
                frame = self._trt_decoder(x_to_dec.contiguous().cuda().half()).squeeze()
            else:
                frame = self.frame_decoder(x_to_dec).squeeze()
        else:
            frame = self.frame_decoder(x_to_dec).squeeze()  # [c,h,w]
        if self.profile_kv:
            e_dec1.record(); torch.cuda.synchronize()
            dec_ms = e_dec0.elapsed_time(e_dec1)
        frame = to_bgr_uint8(frame)
        if self.profile_kv:
            # Throttle using the same counter
            should_print = (self._profile_kv_count <= self.profile_kv_first) or (self._profile_kv_count % self.profile_kv_every == 0)
            if should_print:
                print(f"[KV-Profile] decoder_ms={dec_ms:.2f}ms")
        
        return frame, elapsed_time
    

if __name__ == "__main__":
    pipe = CausvidPipeline()

    # INSERT_YOUR_CODE
    # Simple test: initialize buffers, print their shapes, run a forward pass and print output frame shape

    # Print buffer shapes
    print("history_buffer shape:", pipe.history_buffer.shape if pipe.history_buffer is not None else None)
    print("audio_buffer shape:", pipe.audio_buffer.shape if pipe.audio_buffer is not None else None)
    print("mouse_buffer shape:", pipe.mouse_buffer.shape if pipe.mouse_buffer is not None else None)
    print("button_buffer shape:", pipe.button_buffer.shape if pipe.button_buffer is not None else None)

    # Prepare dummy mouse and button input (matching last dimension of mouse/button buffer)
    mouse_shape = pipe.mouse_buffer.shape[-1] if pipe.mouse_buffer is not None else 2
    button_shape = pipe.button_buffer.shape[-1] if pipe.button_buffer is not None else 11
    import torch

    with torch.no_grad():
        dummy_mouse = torch.zeros(2).bfloat16().cuda()
        dummy_button = torch.zeros(11).bool().cuda()

        # Run a single forward pass
        frame = pipe(dummy_mouse, dummy_button)
        print("Generated frame shape:", frame.shape)
    