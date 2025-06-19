from physicsnemo.utils.profiling import Profiler
import torch
import modelopt.torch.quantization as mtq  # torch-tensorrt model optimizer
import torch_tensorrt
from modelopt.torch.quantization.utils import export_torch_mode

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from owl_wms.utils import load_from_config
from owl_wms.utils.owl_vae_bridge import get_decoder_only

from .profiler import profile_fn, print_results


def profile_torch_compile_tensorrt(world_model, img_dec, audio_dec, dummy, dummy_pred_audio):
    ## Torch Compile with Torch-TensorRT

    tensorrt_backend_args = {
        "use_explicit_typing": True,
        "enabled_precisions": {torch.float32, torch.bfloat16, torch.float16},
        'max_autotune': True,
        'triton.cudagraphs': True,
        'coordinate_descent_tuning': True,
        'debug': True,
    }

    compiled_world_model = torch.compile(world_model, backend="torch_tensorrt", dynamic=False, fullgraph=True, options=tensorrt_backend_args)

    imge_cnn_configs = {
        "device": torch_tensorrt.Device("dla:0", allow_gpu_fallback=True)
        } | tensorrt_backend_args
    compiled_img_dec = torch.compile(img_dec, backend="torch_tensorrt", dynamic=False, fullgraph=True, options=imge_cnn_configs)

    compiled_audio_dec = torch.compile(audio_dec, backend="torch_tensorrt", dynamic=False, fullgraph=True, options=tensorrt_backend_args)

    res_wm = profile_fn(compiled_world_model, dummy)
    print_results(res_wm, "Torch Compile TensorRT - WM")

    res_img = profile_fn(compiled_img_dec, dummy_x[0])
    print_results(res_img, "Torch Compile TensorRT - IMG")

    res_audio = profile_fn(compiled_audio_dec, dummy_pred_audio)
    print_results(res_audio, "Torch Compile TensorRT - AUDIO")


def profile_torch_compile_tensorrt_fp8_torchao(world_model, img_dec, audio_dec, dummy, dummy_pred_audio):
    ## Torch Compile with Torch-TensorRT + FP8 with torchao
    pass


def profile_torch_compile_tensorrt_fp8_tensorrt(world_model, img_dec, audio_dec, dummy, dummy_pred_audio):
    ## Torch Compile with Torch-TensorRT + FP8 with torch-tensorrt
    pass