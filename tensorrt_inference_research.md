# Research: TensorRT Integration for OWL World Model Inference

**Date**: Mon Aug 25 11:14:59 SAST 2025  
**Researcher**: Claude Code  
**Git Commit**: 2fba8c603201c2f83600ab8f1343897407770faa  
**Branch**: main  
**Repository**: owl-wms  

## Research Question

Focus on the inference of the world model. We want to enhance the inference of the model by replacing pytorch with TensorRT. More specifically, we need to do the following: 1) export the pytorch decoder to ONNX, 2) convert the ONNX model to a TensorRT engine, 3) load the TensorRT engine in place of the pytorch model, and 4) replace the pytorch inference calls with TensorRT calls. The main file to modify is 'owl_wms/utils/owl_vae_bridge.py' but be sure to see if any other files need to be edited while keeping in mind that we want to implement extremely focused and succinct edits where only needed.

## Summary

The owl-wms codebase uses VAE decoders for converting latent representations back to perceptual space (video/audio). The current PyTorch-based inference system in `owl_vae_bridge.py` can be enhanced with TensorRT for production inference optimization. Key findings:

1. **Current Architecture**: Two decoder types (video DCAE, audio OobleckVAE) with batched inference functions
2. **Integration Points**: 7 major usage locations across trainers, samplers, and inference pipelines
3. **TensorRT Readiness**: NGC container already includes TensorRT 10.7.0, minimal additional dependencies needed
4. **Complexity Assessment**: Audio decoder (OobleckVAE) is significantly easier to convert than video decoder (DCAE)
5. **Focused Implementation**: Primary changes needed in `owl_vae_bridge.py` with minimal modifications elsewhere

## Detailed Findings

### Current Inference Architecture

**Core Bridge File**: `owl_wms/utils/owl_vae_bridge.py`
- `get_decoder_only()`: Loads PyTorch decoders, removes encoders, returns decoder-only models
- `make_batched_decode_fn()`: Creates batched video decoding function (batch_size=8 default)  
- `make_batched_audio_decode_fn()`: Creates batched audio decoding function (batch_size=8 default)
- **Key Pattern**: All production usage goes through batched decode functions, not direct decoder calls

**Decoder Types Supported**:
1. **DCAE Video Decoder**: ResNet-based 2D convolutional decoder for video frames
2. **AutoencoderDC**: Diffusers-compatible alternative for video
3. **OobleckVAE Audio Decoder**: 1D convolutional decoder for stereo audio

### Usage Patterns Across Codebase

**Major Integration Points**:
1. **Trainers** (6 files): All trainer classes use decoders for evaluation and sampling
   - `owl_wms/trainers/av_trainer.py`: Audio+video decoder setup
   - `owl_wms/trainers/rft_trainer.py`: Video decoder for sampling
   - `owl_wms/trainers/causvid_vid_only.py`: Frame decoding in evaluation
2. **Inference Scripts** (3 files): Production and testing pipelines
   - `inference/test_sampling.py`: Batched decode function usage
   - `inference/causvid_pipeline.py`: Direct decoder call (line 159)
   - `sanity/sampling.py`: Decoder testing and validation
3. **Core Module** (`owl_wms/__init__.py`): `from_pretrained()` integration

**Critical Finding**: Most usage goes through the batched decode functions, but `causvid_pipeline.py` makes direct decoder calls that need special handling.

### Tensor Processing Patterns

**Video Processing**:
- Input: `[batch, n_frames, channels, height, width]` → `[batch*n_frames, channels, height, width]`
- Typical shapes: `[b, n, 128, 8, 8]` → `[b*n, 128, 8, 8]` → decoder → `[b*n, 3, 360, 640]`
- Batch splitting with `batch_size=8` for memory management

**Audio Processing**:  
- Input: `[batch, n_frames, channels]` → transpose → `[batch, channels, n_frames]`
- Typical shapes: `[b, n, 64]` → `[b, 64, n]` → decoder → `[b, 2, audio_length]`
- Similar batch_size=8 processing pattern

**TensorRT Implications**: Fixed batch processing patterns are ideal for static TensorRT engines.

### ONNX Export Complexity Assessment

**OobleckVAE Audio Decoder - RECOMMENDED FIRST TARGET**:
- ✅ Simple 1D convolutions and interpolation operations
- ✅ Straightforward tensor reshaping (transpose only) 
- ✅ Sequential architecture without complex attention mechanisms
- ⚠️ Custom `SnakeBeta` activation needs ONNX-compatible implementation
- ⚠️ `weight_norm` parametrization needs unwrapping

**DCAE Video Decoder - MORE COMPLEX**:
- ❌ Custom landscape/aspect ratio handling operations
- ❌ SANA `pixel_shuffle`/`pixel_unshuffle` with custom channel manipulations
- ❌ Attention mechanisms (`SanaAttn`) with complex reshaping
- ❌ 5D tensor batching requires complex view/reshape operations
- ❌ Multiple custom layer types and einops dependencies

### Environment and Dependencies

**Current State**: Well-positioned for TensorRT integration
- ✅ NGC PyTorch 24.12 container includes TensorRT 10.7.0.23 and Torch-TensorRT 2.6.0a0
- ✅ CUDA 12.6.3 with broad TensorRT compatibility
- ✅ Python 3.12 and cuDNN 9.6.0.74 already compatible

**Minimal Additional Requirements**:
```txt
onnx>=1.17.0
onnxruntime-gpu>=1.19.0
```

**Potential Version Conflict**: PyTorch container has 2.6.0a0 but `owl-vaes/pyproject.toml` specifies `>=2.7.0`

### Current Optimization Infrastructure

**Existing Optimizations**:
- Extensive `torch.compile` usage throughout (`max-autotune`, `dynamic=False` in production)
- MFU (Model FLOPs Utilization) profiling infrastructure
- Hardware-specific optimizations for H200 GPUs

**Missing Components**:
- No existing ONNX export capabilities
- No TensorRT integration code
- No model serialization beyond standard PyTorch checkpoints

## Code References

- `owl_wms/utils/owl_vae_bridge.py:22-38` - get_decoder_only function (main target for modification)
- `owl_wms/utils/owl_vae_bridge.py:41-57` - make_batched_decode_fn (video batching logic)
- `owl_wms/utils/owl_vae_bridge.py:60-75` - make_batched_audio_decode_fn (audio batching logic)
- `inference/causvid_pipeline.py:159` - Direct decoder call requiring special handling
- `owl_wms/__init__.py:from_pretrained` - Integration point for decoder loading
- All trainer files in `owl_wms/trainers/` - Usage of batched decode functions

## Architecture Insights

**Key Design Patterns**:
1. **Decoder Isolation**: Clean separation between encoder and decoder enables targeted optimization
2. **Batched Processing**: Consistent batch_size=8 pattern across all decoders optimizes for TensorRT static engines
3. **Type-Based Loading**: `vae_id` parameter enables different decoder implementations with same interface
4. **Memory Optimization**: bfloat16 precision and GPU-first design throughout

**TensorRT Integration Strategy**:
1. **Minimal Surface Area**: Only `owl_vae_bridge.py` needs core changes
2. **Backwards Compatibility**: Maintain existing function signatures and behavior
3. **Progressive Enhancement**: Start with audio decoder, then video decoder
4. **Engine Management**: TensorRT engines need initialization, warm-up, and memory management

## Implementation Recommendations

### Phase 1: Audio Decoder ONNX Export (Recommended First)
**Primary File**: `owl_wms/utils/owl_vae_bridge.py`
**Changes Needed**:
1. Add ONNX export function for OobleckVAE audio decoder
2. Handle `SnakeBeta` activation replacement
3. Unwrap `weight_norm` parametrizations
4. Test with various input lengths

### Phase 2: TensorRT Engine Integration  
**Primary File**: `owl_wms/utils/owl_vae_bridge.py`
**Changes Needed**:
1. Add TensorRT engine loading function
2. Modify `get_decoder_only()` to optionally return TensorRT engines
3. Update batched decode functions to use TensorRT inference
4. Add engine warm-up and memory management

### Phase 3: Video Decoder Implementation
**Primary File**: `owl_wms/utils/owl_vae_bridge.py` 
**Changes Needed**:
1. Replace custom DCAE operations with ONNX-compatible equivalents
2. Handle complex tensor reshaping for video processing
3. Test with production video dimensions (360x640)

### Focused Edit Strategy

**Core Modifications**:
- `owl_vms/utils/owl_vae_bridge.py`: Add TensorRT engine support to existing functions
- `requirements.txt`: Add ONNX dependencies  
- Minimal config changes to enable TensorRT mode

**Preserved Interfaces**:
- All existing function signatures remain unchanged
- Existing usage patterns in trainers and inference scripts work without modification
- Backwards compatibility with PyTorch-only mode

## Open Questions

1. **Performance Targets**: What latency/throughput improvements are expected?
2. **Engine Persistence**: Should TensorRT engines be cached to disk or rebuilt on each run?
3. **Dynamic vs Static**: Should engines support dynamic batch sizes or optimize for fixed batch_size=8?
4. **Fallback Strategy**: How should the system handle TensorRT engine failures?
5. **Testing Strategy**: What validation approach ensures TensorRT outputs match PyTorch outputs?

## Related Research

This research provides the foundation for implementing TensorRT optimization in the owl-wms inference pipeline. The focused approach targeting `owl_vae_bridge.py` will enable significant performance improvements with minimal codebase disruption.