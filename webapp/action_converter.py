import time
import torch
import asyncio

from webapp.streaming import StreamingConfig
from torch.nn import functional as F

BUTTON_NAMES    = ["W", "A", "S", "D", "LSHIFT", "SPACE", "R", "F", "E", "LMB", "RMB"]
BUTTON_INDICES  = {name: idx for idx, name in enumerate(BUTTON_NAMES)}


def _interpolate(actions: torch.Tensor,
                 empty_action: torch.Tensor,
                 target_length: int) -> torch.Tensor:

    """
    Interpolate actions to target_length.
    If tensor_batch is longer than target_length, subsample.
    If tensor_batch is shorter than target_length, repeat with empty actions.

    Must provide empty_action, which is the action to repeat when the batch is shorter than target_length.
    Must also provide target_length, which is the length to interpolate to.
    Returns:
        actions: [target_length, features]
    """
    num_actions = actions.shape[0]
    
    if num_actions >= target_length:
        # subsample actions if somehow longer than frames_per_batch
        downsampled = torch.arange(0, num_actions, step=(num_actions // target_length))[:target_length]
        return actions[downsampled, :]
    
    # Repeat with empty actions to fill remaining frames
    num_missing_actions = target_length - num_actions
    if num_missing_actions == target_length:
        return empty_action.repeat(target_length, 1)
    
    # NOTE: Repeat last action for the remaining frames
    last_action         = actions[-1, :]
    repeated            = last_action.repeat(num_missing_actions, 1)  # [missing_frames, features]
    
    return torch.cat([actions, repeated], dim=0)  # [target_length, features]


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

    def actions_to_sequence(self, actions: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert list of individual actions to batch tensors.
        
        Args:
            actions: List of (mouse, buttons) tuples
            
        Returns:
            mouse: [sequence_length, 2]
            button: [sequence_length, n_buttons]  
        """
        if not actions:
            # Return empty batch - [0, 2] and [0, n_buttons]
            return (
                torch.zeros(0, 2, device=self.device),
                torch.zeros(0, self.streaming_config.n_buttons, device=self.device)
            )

        mouse  = torch.stack([action[0] for action in actions], dim=0)  # [seq_len, 2]
        button = torch.stack([action[1] for action in actions], dim=0)  # [seq_len, n_buttons]

        return mouse, button
    
    def buttons_to_dict(self, buttons: torch.Tensor) -> dict:
        """Convert buttons tensor to dictionary."""
        return {BUTTON_NAMES[i]: bool(buttons[i]) for i in range(buttons.shape[0])}
    
    def mouse_to_dict(self, mouse: torch.Tensor) -> dict:
        """Convert mouse tensor to dictionary."""
        return {"mouse_x": mouse[0].item(), "mouse_y": mouse[1].item()}


class ActionCollector:
    """Collects real-time actions into 8-frame batches."""
    
    def __init__(self, streaming_config: StreamingConfig):
        self.streaming_config = streaming_config
        self.converter = ActionConverter(streaming_config)
        self.action_queue = asyncio.Queue(maxsize=100)  # Buffer incoming actions

        # an empty action is for one frame only
        self.empty_mouse    = torch.zeros((1, streaming_config.n_mouse_axes),
                                        device=streaming_config.device, dtype=torch.float32)
        self.empty_buttons  = torch.zeros((1, streaming_config.n_buttons),
                                        device=streaming_config.device, dtype=torch.bool)

    async def add_websocket_action(self, ws_message: dict):
        """Add action from WebSocket message."""
        action = self.converter.websocket_to_action(ws_message)
        # Add timestamp to track when action was received
        timestamped_action = (action, time.time())

        await self.action_queue.put(timestamped_action)
    
    async def collect_actions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Collect actions from the UI at whatever rate is sent over by the client. 
        This collects all the frames that have been supplied between the last frame generation and now.
        
        It takes in as many actions as the model generates frames at once. For example, in CausVid, if
         the model generates 4 frames at a time, this function will return [4, 2] and [4, 11]
         for mouse and button actions.
        
        If, for one reason or another, we have <4 actions, we will fill the batch with idle actions.

        Returns:
            mouse:  [X, 2] 
            button: [X, n_buttons]
        """
        real_actions = []
        start_time = time.time()
        
        # First, clear any stale actions from the queue (older than 1 second)
        stale_threshold = start_time - 1.0
        temp_actions = []
        stale_count = 0
        
        # Drain the queue and filter out stale actions
        while not self.action_queue.empty():
            try:
                timestamped_action = self.action_queue.get_nowait()
                action, timestamp = timestamped_action
                if timestamp >= stale_threshold:
                    temp_actions.append(action)
                else:
                    stale_count += 1
            except asyncio.QueueEmpty:
                break
        
        # Re-add fresh actions to the queue
        for action in temp_actions:
            try:
                await self.action_queue.put((action, time.time()))
            except asyncio.QueueFull:
                break  # Skip if queue is full
        
        # Now collect actions for the current batch
        while start_time + self.streaming_config.batch_duration > time.time():
            try:
                timeout = max(0.01, self.streaming_config.batch_duration - (time.time() - start_time))
                timestamped_action = await asyncio.wait_for(self.action_queue.get(), timeout=timeout)
                action, timestamp = timestamped_action
                real_actions.append(action)
            except asyncio.TimeoutError:
                pass

        # Convert real actions to batch tensors
        mouse, button   = self.converter.actions_to_sequence(real_actions)
        mouse           = _interpolate(mouse, empty_action=self.empty_mouse,
                                              target_length=self.streaming_config.frames_per_batch)
        button          = _interpolate(button,empty_action=self.empty_buttons,
                                              target_length=self.streaming_config.frames_per_batch)
        
        return mouse, button
