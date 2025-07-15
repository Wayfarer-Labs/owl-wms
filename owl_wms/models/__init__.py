def get_model_cls(model_id):
    if model_id == "game_rft":
        from .gamerft import GameRFT
        return GameRFT
    if model_id == "game_rft_audio":
        from .gamerft_audio import GameRFTAudio
        return GameRFTAudio




