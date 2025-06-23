from physicsnemo.utils.profiling import Profiler
import torch

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from owl_wms.utils import load_from_config
from owl_wms.utils.owl_vae_bridge import get_decoder_only

from .profiler import profile_fn, print_results

# torchao quantization
import torchao
from torchao.quantization.autoquant import OTHER_AUTOQUANT_CLASS_LIST, ALL_AUTOQUANT_CLASS_LIST  # OTHER contains only FLOAT8
# autoquant uses int8 by default with additional support for int4, not FP8
from torchao.quantization import (
    quantize_,
    Float8WeightOnlyConfig,
    Float8DynamicActivationFloat8WeightConfig,
    PerTensor,
    PerRow,
)


def profile_torch_compile_inductor(world_model, img_dec, audio_dec, dummy, dummy_pred_audio):
    ## Torch Compile with Inductor

    compiled_world_model = torch.compile(world_model, mode='max-autotune', dynamic=False, fullgraph=True)
    compiled_img_dec = torch.compile(img_dec, mode='max-autotune', dynamic=False, fullgraph=True)
    compiled_audio_dec = torch.compile(audio_dec, mode='max-autotune', dynamic=False, fullgraph=True)

    res_wm = profile_fn(compiled_world_model, dummy)
    print_results(res_wm, "Torch Compile - WM")

    res_img = profile_fn(compiled_img_dec, dummy[0][0])
    print_results(res_img, "Torch Compile - IMG")

    res_audio = profile_fn(compiled_audio_dec, dummy_pred_audio)
    print_results(res_audio, "Torch Compile - AUDIO")


def profile_torch_compile_inductor_fp8_torchao(world_model, img_dec, audio_dec, dummy, dummy_pred_audio):
    ## Torch Compile with Inductor + FP8 with torchao

    compiled_world_model = torch.compile(world_model, mode='max-autotune', dynamic=False, fullgraph=True)
    compiled_img_dec = torch.compile(img_dec, mode='max-autotune', dynamic=False, fullgraph=True)
    compiled_audio_dec = torch.compile(audio_dec, mode='max-autotune', dynamic=False, fullgraph=True)

    compiled_world_model = torchao.quantization.autoquant(
        compiled_world_model,
        example_input=dummy,
        qtensor_class_list=ALL_AUTOQUANT_CLASS_LIST,
        set_inductor_config=False
    )

    compiled_img_dec = torchao.quantization.autoquant(
        compiled_img_dec,
        example_input=dummy[0][0],
        qtensor_class_list=ALL_AUTOQUANT_CLASS_LIST,
        set_inductor_config=False
    )

    compiled_audio_dec = torchao.quantization.autoquant(
        compiled_audio_dec,
        example_input=dummy_pred_audio,
        qtensor_class_list=ALL_AUTOQUANT_CLASS_LIST,
        set_inductor_config=False
    )

    res_wm = profile_fn(compiled_world_model, dummy)
    print_results(res_wm, "Torch Compile + TorchAO AutoQuant - WM")

    res_img = profile_fn(compiled_img_dec, dummy[0][0])
    print_results(res_img, "Torch Compile + TorchAO AutoQuant - IMG")

    res_audio = profile_fn(compiled_audio_dec, dummy_pred_audio)
    print_results(res_audio, "Torch Compile + TorchAO AutoQuant - AUDIO")
