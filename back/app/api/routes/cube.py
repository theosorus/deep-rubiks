# Fastapi
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Request

from models.moves import NOTATION_TO_MOVE
from models.move_request import MoveRequest


cube_router = APIRouter()

@cube_router.get("/get-cube")
async def get_cube(request: Request):
    """
    Get the current state of the cube
    """
    cube_json = request.app.state.cube.__dict__()
    return cube_json


@cube_router.get("/moves")
async def get_cube(request: Request):
    moves = list(NOTATION_TO_MOVE.keys())
    print(f"Available moves: {moves}")
    return moves


@cube_router.post("/rotate")
async def make_move(request: Request, move_request: MoveRequest):
    """
    Make a move on the cube based on the notation
    """
    received_move = move_request.move
    cube = request.app.state.cube
    print(f"Received move: {received_move}")
    move_class = NOTATION_TO_MOVE.get(received_move.upper())
    print(f"Move class found: {move_class}")
    
    if not move_class:
        return JSONResponse(status_code=400, content={"error": "Invalid move notation"})
    
    move_instance = move_class()
    move_instance.make_move_on_cube(cube)
    
    print(f"Cube state after move:")
    print(cube)
    
    return JSONResponse(
        status_code=200,
        content={
            "message": f"Move {received_move} applied successfully"})
    
    
@cube_router.get("/reset")
async def reset_cube(request: Request):
    cube = request.app.state.cube
    cube.reset_cube()
    return JSONResponse(
        status_code=200,
        content={
            "message": "Cube reset successfully",
        })
