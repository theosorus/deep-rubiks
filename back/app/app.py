import logging
from contextlib import asynccontextmanager
import asyncio
import uvicorn
import uvloop
import os

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from core.cube import Cube
from solver.rubiks_cube_adapter import RubiksCubeAdapter
from solver.utils import load_model
from solver.astar import AStarSolver
from api.main_router import api_router

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup")
    
    # Cube creation
    app.state.cube = Cube()
    print(app.state.cube)
    logger.info("Cube created and stored in app state")
    
    artifacts = load_model("./output/rubiks_cube_solver_small.pth")
    adapter = RubiksCubeAdapter()
    app.state.solver = AStarSolver(artifacts.net, adapter)
    # DEVICE 
    
    yield
    logger.info("Application shutdown")


app = FastAPI(
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],                     
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

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