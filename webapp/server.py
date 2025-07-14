import os
from dotenv import load_dotenv ; load_dotenv()

from contextlib             import asynccontextmanager
from fastapi                import FastAPI, WebSocket
from fastapi.staticfiles    import StaticFiles
from fastapi.responses      import FileResponse

from webapp.streaming       import StreamingFrameGenerator
from webapp.user_session    import UserGameSession
from webapp.utils.configs   import WebappConfig


DEBUG = False 

# -- lifespan
config: WebappConfig    = None
webapp_config_path      = "./webapp/webapp_config.yaml" ; assert os.path.exists(webapp_config_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, DEBUG
    config = WebappConfig.from_yaml(webapp_config_path)
    yield
    config = None


def run():
    """Create and configure the FastAPI app with routes."""
    app = FastAPI(lifespan=lifespan)
    
    @app.get("/")
    async def read_root():
        """Serve the main game page."""
        return FileResponse("webapp/static/index.html")

    @app.websocket("/ws/game")
    async def websocket_endpoint(websocket: WebSocket):
        global DEBUG
        await websocket.accept()
        
        # Create streaming session for this user
        frame_generator = StreamingFrameGenerator(streaming_config=config.stream_config,
                                                  run_config=config.run_config,
                                                  debug=DEBUG)
        session = UserGameSession(frame_generator)
        await session.run_session(websocket)
    
    app.mount("/assets", StaticFiles(directory="webapp/static"), name="assets")
    return app


def main():
    global DEBUG
    import argparse
    import uvicorn

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-debug", action="store_true", default=True, help="Disable debug mode")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    

    if args.debug:
        DEBUG = True
    elif args.no_debug:
        DEBUG = False
    # Otherwise keep the default value (True)

    # Create app AFTER setting DEBUG
    app = run()

    print("🚀 Starting OWL-WMS FastAPI Server...")
    print("📡 WebSocket endpoint: ws://localhost:8000/ws/game")
    print("🌐 Access via: http://localhost:8000")
    print("🔄 Auto-reload enabled for development")
    print("🔄 DEBUG is set to:", DEBUG)
    print("🔄 PORT is set to:", args.port)

    uvicorn.run(
        app,  # Pass the app object directly instead of module string
        host="0.0.0.0",  # Allow external connections
        port=args.port,
        reload=False,    # Can't use reload with app object
        log_level="info"
    )


if __name__ == "__main__":
    main()
