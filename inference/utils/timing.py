import torch
from typing import Callable, List, Tuple, Any


def time_function(
    fn: Callable,
    *inputs,
    n_warmup_calls: int = 10,
    n_eval_calls: int = 100
) -> Tuple[float, float, float, float, float, float]:
    """
    Time a function using CUDA events for accurate GPU timing.

    Args:
        fn: The function to time
        *inputs: Input arguments to the function. Tensor inputs will be randomized
                 for each call.
        n_warmup_calls: Number of warmup calls before timing
        n_eval_calls: Number of evaluation calls to time

    Returns:
        Tuple of (min_latency_ms, max_latency_ms, avg_latency_ms,
                 min_fps, max_fps, avg_fps)
    """
    torch.cuda.synchronize()

    # Warmup
    for _ in range(n_warmup_calls):
        randomized_inputs = [
            torch.randn_like(inp) if (isinstance(inp, torch.Tensor) and not inp.dtype in (torch.long, torch.int, torch.int64, torch.int32, torch.int16, torch.int8, torch.bool)) else inp
            for inp in inputs
        ]
        _ = fn(*randomized_inputs)
        torch.cuda.synchronize()

    # Timing runs
    latencies = []
    for _ in range(n_eval_calls):
        randomized_inputs = [
            torch.randn_like(inp) if isinstance(inp, torch.Tensor) and not inp.dtype in (torch.long, torch.int, torch.int64, torch.int32, torch.int16, torch.int8, torch.bool) else inp
            for inp in inputs
        ]

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = fn(*randomized_inputs)
        end_event.record()

        torch.cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event)
        latencies.append(latency_ms)

    # Calculate statistics
    min_latency = min(latencies)
    max_latency = max(latencies)
    avg_latency = sum(latencies) / len(latencies)

    # Convert to FPS (frames per second)
    # FPS = 1000ms / latency_ms
    min_fps = 1000.0 / max_latency  # max latency = min fps
    max_fps = 1000.0 / min_latency  # min latency = max fps
    avg_fps = 1000.0 / avg_latency

    return min_latency, max_latency, avg_latency, min_fps, max_fps, avg_fps