from .gamerft import GameRFT

def get_model_cls(model_id):
    if model_id == "game_rft":
        return GameRFT




