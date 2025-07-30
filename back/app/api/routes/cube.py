# Fastapi
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Request


cube_router = APIRouter()

@cube_router.get("/get-cube")
async def get_cube(request: Request):
    """
    Get the current state of the cube
    """
    cube_json = request.app.state.cube.__dict__()
    return cube_json