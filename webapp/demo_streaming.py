import asyncio
import termcolor
import torch
import json

from webapp.streaming import StreamingConfig, StreamingFrameGenerator
from owl_wms.configs import TrainingConfig, WindowSamplingConfig, TransformerConfig as ModelConfig
from webapp.models import load_models
from webapp.user_session import UserGameSession

CFG_PATH = 'checkpoints/wm/dcae_hf_cod/basic.yml'

async def demo_streaming_session():
    """Demo the streaming system."""
    streaming_config = StreamingConfig(
        fps=20,
        frames_per_batch=8,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    sampling_config = WindowSamplingConfig()
    encoder, decoder, train_run_config = load_models(device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True)
    model_config: ModelConfig = train_run_config.model
    training_config: TrainingConfig = train_run_config.train

    frame_generator = StreamingFrameGenerator(encoder, decoder,
                                              streaming_config, model_config, training_config, sampling_config)
    session = UserGameSession(frame_generator)

    class MockWebSocket:
        async def __aiter__(self):
            # Generate dummy actions
            while True:
                action = {
                    "mouse_x": 0.1,
                    "mouse_y": 0.0,
                    "W": True,
                    "LMB": False
                }
                yield json.dumps(action)
                await asyncio.sleep(0.05)  # 20 FPS
        
        async def send(self, data):
            print(f"📤 Sent: {data}")
    
    mock_ws = MockWebSocket()
    await session.run_session(mock_ws)


if __name__ == "__main__":
    print(termcolor.colored("OWL-WMS Real-time Streaming System", "green"))
    print(termcolor.colored("=" * 50, "green"))
    
    asyncio.run(demo_streaming_session())
