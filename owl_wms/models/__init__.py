def get_model_cls(model_id):
    if model_id == "game_rft":
        from .gamerft import GameRFT
        return GameRFT
    if model_id == "game_rft_shortcut":
        from .gamerft_shortcut import ShortcutGameRFT
        return ShortcutGameRFT
    if model_id == "shortcut_2":
        from .gamerft_shortcut_simple import ShortcutGameRFT
        return ShortcutGameRFT
    if model_id == "game_rft_audio":
        from .gamerft_audio import GameRFT
        return GameRFT




