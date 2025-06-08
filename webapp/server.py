import os
from torch import nn

from contextlib             import asynccontextmanager
from fastapi                import FastAPI, WebSocket
from fastapi.staticfiles    import StaticFiles
from fastapi.responses      import FileResponse

from webapp.utils.models    import load_models
from webapp.streaming       import StreamingFrameGenerator
from webapp.user_session    import UserGameSession
from webapp.utils.configs   import WebappConfig


DEBUG = True # for funsies

# -- lifespan
encoder: nn.Module      = None
decoder: nn.Module      = None
config: WebappConfig    = None
webapp_config_path      = "./configs/webapp/config.yaml" ; assert os.path.exists(webapp_config_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, decoder, config
    config = WebappConfig.from_yaml(webapp_config_path)
    if not DEBUG:
        encoder, decoder, _ = load_models(
            checkpoint_path=config.model_checkpoint_path,
            config_path=config.run_config_path,
            device=config.device, verbose=True,
        )

    yield
    encoder, decoder, config = None, None, None


app = FastAPI(lifespan=lifespan)
app.mount("/assets", StaticFiles(directory="webapp/static"), name="assets")


@app.get("/")
async def read_root():
    """Serve the main game page."""
    return FileResponse("webapp/static/index.html")


@app.websocket("/ws/game")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create streaming session for this user
    frame_generator = StreamingFrameGenerator(encoder, decoder,
                                              streaming_config=config.stream_config,
                                              model_config=config.run_config.model,
                                              train_config=config.run_config.train,
                                              sampling_config=config.sampling_config,
                                              debug=DEBUG)
    session = UserGameSession(frame_generator)
    
    # Run the session (your existing code!)
    await session.run_session(websocket)


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting OWL-WMS FastAPI Server...")
    print("📡 WebSocket endpoint: ws://localhost:8000/ws/game")
    print("🌐 Access via: http://localhost:8000")
    print("🔄 Auto-reload enabled for development")
    
    uvicorn.run(
        "webapp.server:app",
        host="0.0.0.0",  # Allow external connections
        port=8000,
        reload=True,     # Auto-reload on file changes
        log_level="info"
    )
