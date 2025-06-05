import torch
import asyncio
import time

from webapp.streaming import StreamingConfig

BUTTON_NAMES    = ["W", "A", "S", "D", "LSHIFT", "SPACE", "R", "F", "E", "LMB", "RMB"]
BUTTON_INDICES  = {name: idx for idx, name in enumerate(BUTTON_NAMES)}

class ActionConverter:
    """Converts WebSocket messages to model tensor format."""

    def __init__(self, streaming_config: StreamingConfig):
        self.streaming_config = streaming_config
        self.device = streaming_config.device
        
    def websocket_to_action(self, ws_message: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert WebSocket message to action tensors.
        
        Expected ws_message format:
        {
            "mouse_x": 0.1,    # Mouse movement [-1, 1]
            "mouse_y": -0.05,
            "W": true,         # Button states
            "LMB": false,
            # ... other buttons
        }
        
        Returns:
            mouse: [2] tensor  
            buttons: [n_buttons] tensor
        """
        # Extract mouse movement
        mouse_x = ws_message.get("mouse_x", 0.0)
        mouse_y = ws_message.get("mouse_y", 0.0)
        
        # Clamp to valid range
        mouse_x = max(min(mouse_x, self.streaming_config.mouse_range[1]), self.streaming_config.mouse_range[0])
        mouse_y = max(min(mouse_y, self.streaming_config.mouse_range[1]), self.streaming_config.mouse_range[0])
        
        mouse = torch.tensor([mouse_x, mouse_y], device=self.device, dtype=torch.float32)
        
        # Extract button states
        button_states = torch.zeros(self.streaming_config.n_buttons, device=self.device, dtype=torch.float32)
        for button_name, idx in BUTTON_INDICES.items():
            if button_name in ws_message:
                button_states[idx] = 1.0 if ws_message[button_name] else 0.0
        
        return mouse, button_states

    def actions_to_batch(self, actions: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert list of individual actions to batch tensors.
        
        Args:
            actions: List of (mouse, buttons) tuples
            
        Returns:
            mouse_batch: [batch_size, sequence_length, 2]
            button_batch: [batch_size, sequence_length, n_buttons]  
        """
        if not actions:
            # Return empty batch
            return (
                torch.zeros(1, 0, 2, device=self.device),
                torch.zeros(1, 0, self.streaming_config.n_buttons, device=self.device)
            )
        
        mouse_list  = [action[0] for action in actions]
        button_list = [action[1] for action in actions]
        
        # Stack into sequences and add batch dimension
        mouse_batch = torch.stack(mouse_list, dim=0).unsqueeze(0)  # [1, seq_len, 2]
        button_batch = torch.stack(button_list, dim=0).unsqueeze(0)  # [1, seq_len, n_buttons]
        
        return mouse_batch, button_batch


class ActionCollector:
    """Collects real-time actions into 8-frame batches."""
    
    def __init__(self, streaming_config: StreamingConfig):
        self.streaming_config = streaming_config
        self.converter = ActionConverter(streaming_config)
        self.action_queue = asyncio.Queue(maxsize=100)  # Buffer incoming actions
        self.current_batch = []
        
    async def add_websocket_action(self, ws_message: dict):
        """Add action from WebSocket message."""
        action = self.converter.websocket_to_action(ws_message)
        await self.action_queue.put(action)
    
    async def collect_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Collect actions for one batch (8 frames worth).
        
        Returns action tensors ready for model inference.
        """
        actions = []
        batch_duration = self.streaming_config.batch_duration
        start_time = time.time()
        
        # Collect actions for the batch duration (400ms for 8 frames)
        while len(actions) < self.streaming_config.frames_per_batch:
            try:
                # Wait for next action, but don't wait forever
                timeout = max(0.01, batch_duration - (time.time() - start_time))
                action = await asyncio.wait_for(self.action_queue.get(), timeout=timeout)
                actions.append(action)
            except asyncio.TimeoutError:
                # If no action received, repeat the last action or use idle
                if actions:
                    actions.append(actions[-1])  # Repeat last action
                else:
                    # Generate idle action
                    idle_mouse = torch.zeros(2, device=self.streaming_config.device)
                    idle_buttons = torch.zeros(self.streaming_config.n_buttons, device=self.streaming_config.device)
                    actions.append((idle_mouse, idle_buttons))
        
        # Convert to batch tensors
        return self.converter.actions_to_batch(actions)
