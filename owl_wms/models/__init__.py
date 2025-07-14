def get_model_cls(model_id):
    if model_id == "game_rft":
        from .gamerft import GameRFT
        return GameRFT
<<<<<<< HEAD
    if model_id == "game_rft_shortcut":
        from .gamerft_shortcut import ShortcutGameRFT
        return ShortcutGameRFT
    if model_id == "causal_game_rft":
        from .causal_gamerft import CausalGameRFT
        return CausalGameRFT
    if model_id == "shortcut_2":
        from .gamerft_shortcut_simple import ShortcutGameRFT
        return ShortcutGameRFT
=======
>>>>>>> uncond
    if model_id == "game_rft_audio":
        from .gamerft_audio import GameRFTAudio
        return GameRFTAudio




