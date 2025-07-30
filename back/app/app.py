import logging
from contextlib import asynccontextmanager
import asyncio
import uvicorn
import uvloop
import os

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from models.cube import Cube
from api.main_router import api_router

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup")
    
    # Cube creation
    app.state.cube = Cube()
    app.state.cube.shuffle_cube(min_move=100, max_move=500)
    logger.info("Cube created and stored in app state")
    
    yield
    
    # Cleanup if needed
    logger.info("Application shutdown")


app = FastAPI(
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "PUT"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

# Set uvloop as the default event loop policy
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level='info',
        workers=os.cpu_count() + 1,
        reload=True,
        loop="uvloop",
    )