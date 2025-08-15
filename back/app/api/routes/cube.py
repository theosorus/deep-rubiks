# Fastapi
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Request

from core.moves import AVAILABLE_MOVES,MOVE_CLASSES
from models.move_request import MoveRequest


cube_router = APIRouter()

@cube_router.get("/get-cube")
async def get_cube(request: Request):
    """
    Get the current state of the cube
    """
    cube_json = request.app.state.cube.__dict__()
    print(cube_json)
    return cube_json


@cube_router.get("/moves")
async def get_cube(request: Request):
    print(f"Available moves: {AVAILABLE_MOVES}")
    return AVAILABLE_MOVES


@cube_router.post("/rotate")
async def make_move(request: Request, move_request: MoveRequest):
    """
    Make a move on the cube based on the notation
    """
    received_move = move_request.move
    cube = request.app.state.cube
    print(f"Received move: {received_move}")
    move_class = MOVE_CLASSES.get(received_move.upper())
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
    print(cube)
    return JSONResponse(
        status_code=200,
        content={
            "message": "Cube reset successfully",
        })
    
    
@cube_router.get("/shuffle")
async def shuffle_cube(request: Request, nb_moves: int):
    """
    Shuffle the cube with a random number of moves (nb_moves passed as query param)
    """
    cube = request.app.state.cube
    cube.shuffle_cube(min_move=nb_moves, max_move=nb_moves)
    print(cube)
    return JSONResponse(
        status_code=200,
        content={
            "message": f"Cube shuffled with {nb_moves} moves",
        })
