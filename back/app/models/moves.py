from abc import ABC, abstractmethod
import numpy as np
import random

class Move(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def make_move_on_cube(self, cube: 'Cube') -> None:
        """
        Abstract method to make a move on the cube.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

class MoveUPLeft(Move):
    """Move the upper left corner of the cube (U face clockwise)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate top face (face 0) clockwise
        cube.state[0] = np.rot90(cube.state[0], -1)
        
        # Rotate edges: front -> right -> back -> left -> front
        temp = cube.state[1][0].copy()  # front top row
        cube.state[1][0] = cube.state[4][0]  # left top row -> front top row
        cube.state[4][0] = cube.state[2][0]  # back top row -> left top row
        cube.state[2][0] = cube.state[3][0]  # right top row -> back top row
        cube.state[3][0] = temp  # front top row -> right top row

class MoveUpRight(Move):
    """Move the upper right corner of the cube (U face counter-clockwise)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate top face (face 0) counter-clockwise
        cube.state[0] = np.rot90(cube.state[0], 1)
        
        # Rotate edges: front -> left -> back -> right -> front
        temp = cube.state[1][0].copy()  # front top row
        cube.state[1][0] = cube.state[3][0]  # right top row -> front top row
        cube.state[3][0] = cube.state[2][0]  # back top row -> right top row
        cube.state[2][0] = cube.state[4][0]  # left top row -> back top row
        cube.state[4][0] = temp  # front top row -> left top row

class MoveMiddleLeft(Move):
    """Move the middle left side of the cube (M slice like L)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate middle slice: front -> top -> back -> bottom -> front
        temp = cube.state[1][:, 1].copy()  # front middle column
        cube.state[1][:, 1] = cube.state[5][:, 1]  # bottom middle column -> front
        cube.state[5][:, 1] = cube.state[2][:, 1][::-1]  # back middle column (reversed) -> bottom
        cube.state[2][:, 1] = cube.state[0][:, 1][::-1]  # top middle column (reversed) -> back
        cube.state[0][:, 1] = temp  # front middle column -> top

class MoveMiddleRight(Move):
    """Move the middle right side of the cube (M slice opposite direction)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate middle slice: front -> bottom -> back -> top -> front
        temp = cube.state[1][:, 1].copy()  # front middle column
        cube.state[1][:, 1] = cube.state[0][:, 1]  # top middle column -> front
        cube.state[0][:, 1] = cube.state[2][:, 1][::-1]  # back middle column (reversed) -> top
        cube.state[2][:, 1] = cube.state[5][:, 1][::-1]  # bottom middle column (reversed) -> back
        cube.state[5][:, 1] = temp  # front middle column -> bottom

class MoveBottomLeft(Move):
    """Move the bottom left corner of the cube (D face counter-clockwise)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate bottom face (face 5) counter-clockwise
        cube.state[5] = np.rot90(cube.state[5], 1)
        
        # Rotate edges: front -> left -> back -> right -> front
        temp = cube.state[1][2].copy()  # front bottom row
        cube.state[1][2] = cube.state[3][2]  # right bottom row -> front bottom row
        cube.state[3][2] = cube.state[2][2]  # back bottom row -> right bottom row
        cube.state[2][2] = cube.state[4][2]  # left bottom row -> back bottom row
        cube.state[4][2] = temp  # front bottom row -> left bottom row

class MoveBottomRight(Move):
    """Move the bottom right corner of the cube (D face clockwise)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate bottom face (face 5) clockwise
        cube.state[5] = np.rot90(cube.state[5], -1)
        
        # Rotate edges: front -> right -> back -> left -> front
        temp = cube.state[1][2].copy()  # front bottom row
        cube.state[1][2] = cube.state[4][2]  # left bottom row -> front bottom row
        cube.state[4][2] = cube.state[2][2]  # back bottom row -> left bottom row
        cube.state[2][2] = cube.state[3][2]  # right bottom row -> back bottom row
        cube.state[3][2] = temp  # front bottom row -> right bottom row

class MoveLeftUp(Move):
    """Move the left side of the cube upwards (L face clockwise)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate left face (face 4) clockwise
        cube.state[4] = np.rot90(cube.state[4], -1)
        
        # Rotate edges: front -> top -> back -> bottom -> front
        temp = cube.state[1][:, 0].copy()  # front left column
        cube.state[1][:, 0] = cube.state[5][:, 0]  # bottom left column -> front
        cube.state[5][:, 0] = cube.state[2][:, 2][::-1]  # back right column (reversed) -> bottom
        cube.state[2][:, 2] = cube.state[0][:, 0][::-1]  # top left column (reversed) -> back
        cube.state[0][:, 0] = temp  # front left column -> top

class MoveLeftDown(Move):
    """Move the left side of the cube downwards (L face counter-clockwise)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate left face (face 4) counter-clockwise
        cube.state[4] = np.rot90(cube.state[4], 1)
        
        # Rotate edges: front -> bottom -> back -> top -> front
        temp = cube.state[1][:, 0].copy()  # front left column
        cube.state[1][:, 0] = cube.state[0][:, 0]  # top left column -> front
        cube.state[0][:, 0] = cube.state[2][:, 2][::-1]  # back right column (reversed) -> top
        cube.state[2][:, 2] = cube.state[5][:, 0][::-1]  # bottom left column (reversed) -> back
        cube.state[5][:, 0] = temp  # front left column -> bottom

class MoveMiddleUp(Move):
    """Move the middle layer of the cube upwards (E slice opposite direction)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate middle horizontal slice: front -> left -> back -> right -> front
        temp = cube.state[1][1].copy()  # front middle row
        cube.state[1][1] = cube.state[3][1]  # right middle row -> front
        cube.state[3][1] = cube.state[2][1]  # back middle row -> right
        cube.state[2][1] = cube.state[4][1]  # left middle row -> back
        cube.state[4][1] = temp  # front middle row -> left

class MoveMiddleDown(Move):
    """Move the middle layer of the cube downwards (E slice like D)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate middle horizontal slice: front -> right -> back -> left -> front
        temp = cube.state[1][1].copy()  # front middle row
        cube.state[1][1] = cube.state[4][1]  # left middle row -> front
        cube.state[4][1] = cube.state[2][1]  # back middle row -> left
        cube.state[2][1] = cube.state[3][1]  # right middle row -> back
        cube.state[3][1] = temp  # front middle row -> right

class MoveRightUp(Move):
    """Move the right side of the cube upwards (R face counter-clockwise)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate right face (face 3) counter-clockwise
        cube.state[3] = np.rot90(cube.state[3], 1)
        
        # Rotate edges: front -> bottom -> back -> top -> front
        temp = cube.state[1][:, 2].copy()  # front right column
        cube.state[1][:, 2] = cube.state[5][:, 2]  # bottom right column -> front
        cube.state[5][:, 2] = cube.state[2][:, 0][::-1]  # back left column (reversed) -> bottom
        cube.state[2][:, 0] = cube.state[0][:, 2][::-1]  # top right column (reversed) -> back
        cube.state[0][:, 2] = temp  # front right column -> top

class MoveRightDown(Move):
    """Move the right side of the cube downwards (R face clockwise)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate right face (face 3) clockwise
        cube.state[3] = np.rot90(cube.state[3], -1)
        
        # Rotate edges: front -> top -> back -> bottom -> front
        temp = cube.state[1][:, 2].copy()  # front right column
        cube.state[1][:, 2] = cube.state[0][:, 2]  # top right column -> front
        cube.state[0][:, 2] = cube.state[2][:, 0][::-1]  # back left column (reversed) -> top
        cube.state[2][:, 0] = cube.state[5][:, 2][::-1]  # bottom right column (reversed) -> back
        cube.state[5][:, 2] = temp  # front right column -> bottom

class MoveBackLeft(Move):
    """Move the back left side of the cube (B face counter-clockwise)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate back face (face 2) counter-clockwise
        cube.state[2] = np.rot90(cube.state[2], 1)
        
        # Rotate edges: top -> right -> bottom -> left -> top
        temp = cube.state[0][0].copy()  # top back row
        cube.state[0][0] = cube.state[3][:, 2]  # right back column -> top back row
        cube.state[3][:, 2] = cube.state[5][2][::-1]  # bottom back row (reversed) -> right back column
        cube.state[5][2] = cube.state[4][:, 0]  # left back column -> bottom back row
        cube.state[4][:, 0] = temp[::-1]  # top back row (reversed) -> left back column

class MoveBackRight(Move):
    """Move the back right side of the cube (B face clockwise)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate back face (face 2) clockwise
        cube.state[2] = np.rot90(cube.state[2], -1)
        
        # Rotate edges: top -> left -> bottom -> right -> top
        temp = cube.state[0][0].copy()  # top back row
        cube.state[0][0] = cube.state[4][:, 0][::-1]  # left back column (reversed) -> top back row
        cube.state[4][:, 0] = cube.state[5][2]  # bottom back row -> left back column
        cube.state[5][2] = cube.state[3][:, 2][::-1]  # right back column (reversed) -> bottom back row
        cube.state[3][:, 2] = temp  # top back row -> right back column

class MoveSliceLeft(Move):
    """Move a slice of the cube to the left (S slice like F)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate middle vertical slice: top -> right -> bottom -> left -> top
        temp = cube.state[0][1].copy()  # top middle row
        cube.state[0][1] = cube.state[4][:, 1][::-1]  # left middle column (reversed) -> top middle row
        cube.state[4][:, 1] = cube.state[5][1]  # bottom middle row -> left middle column
        cube.state[5][1] = cube.state[3][:, 1][::-1]  # right middle column (reversed) -> bottom middle row
        cube.state[3][:, 1] = temp  # top middle row -> right middle column

class MoveSliceRight(Move):
    """Move a slice of the cube to the right (S slice opposite direction)"""
    def make_move_on_cube(self, cube: 'Cube') -> None:
        # Rotate middle vertical slice: top -> left -> bottom -> right -> top
        temp = cube.state[0][1].copy()  # top middle row
        cube.state[0][1] = cube.state[3][:, 1]  # right middle column -> top middle row
        cube.state[3][:, 1] = cube.state[5][1][::-1]  # bottom middle row (reversed) -> right middle column
        cube.state[5][1] = cube.state[4][:, 1]  # left middle column -> bottom middle row
        cube.state[4][:, 1] = temp[::-1]  # top middle row (reversed) -> left middle column

MOVES_MAP = {
    1: MoveUPLeft,
    2: MoveUpRight,
    3: MoveMiddleLeft,
    4: MoveMiddleRight,
    5: MoveBottomLeft,
    6: MoveBottomRight,
    7: MoveLeftUp,
    8: MoveLeftDown,
    9: MoveMiddleUp,
    10: MoveMiddleDown,
    11: MoveRightUp,
    12: MoveRightDown,
    13: MoveBackLeft,
    14: MoveBackRight,
    15: MoveSliceLeft,
    16: MoveSliceRight
}

def get_random_move() -> Move:
    """Get a random move from the MOVES_MAP"""
    return random.choice(list(MOVES_MAP.values()))()

def get_random_moves(n: int) -> list[Move]:
    """Get a list of random moves"""
    return [random.choice(list(MOVES_MAP.values()))() for _ in range(n)]