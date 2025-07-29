from abc import ABC, abstractmethod
from models.cube import Cube
import numpy as np
import random 



class Move(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def make_move_on_cube(self, cube : Cube) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")


class MoveUL(Move):
    def make_move_on_cube(self, cube: Cube) -> None:
        cube.state = np.rot90(cube.state, k=1, axes=(0, 1))

class MoveUR(Move):
    def make_move_on_cube(self, cube: Cube) -> None:
        cube.state = np.rot90(cube.state, k=-1, axes=(0, 1))

class MoveML(Move):
    def make_move_on_cube(self, cube: Cube) -> None:
        cube.state = np.rot90(cube.state, k=1, axes=(0, 2))

class MoveMR(Move):
    def make_move_on_cube(self, cube: Cube) -> None:
        cube.state = np.rot90(cube.state, k=-1, axes=(0, 2))

class MoveDL(Move):
    def make_move_on_cube(self, cube: Cube) -> None:
        cube.state = np.rot90(cube.state, k=1, axes=(1, 2))

class MoveDR(Move):
    def make_move_on_cube(self, cube: Cube) -> None:
        cube.state = np.rot90(cube.state, k=-1, axes=(1, 2))

class MoveLU(Move):
    def make_move_on_cube(self, cube: Cube) -> None:
        cube.state = np.rot90(cube.state, k=1, axes=(1, 0))

class MoveLD(Move):
    def make_move_on_cube(self, cube: Cube) -> None:
        cube.state = np.rot90(cube.state, k=-1, axes=(1, 0))

class MoveMU(Move):
    def make_move_on_cube(self, cube: Cube) -> None:
        cube.state = np.rot90(cube.state, k=1, axes=(2, 0))

class MoveMD(Move):
    def make_move_on_cube(self, cube: Cube) -> None:
        cube.state = np.rot90(cube.state, k=-1, axes=(2, 0))

class MoveRU(Move):
    def make_move_on_cube(self, cube: Cube) -> None:
        cube.state = np.rot90(cube.state, k=1, axes=(2, 1))

class MoveRD(Move):
    def make_move_on_cube(self, cube: Cube) -> None:
        cube.state = np.rot90(cube.state, k=-1, axes=(2, 1))


MOVES_MAP = {
    1 : MoveUR,
    2 : MoveUL,
    3 : MoveML,
    4 : MoveMR,
    5 : MoveDL,
    6 : MoveDR,
    7 : MoveLU,
    8 : MoveLD,
    9 : MoveMU,
    10 : MoveMD,
    11 : MoveRU,
    12 : MoveRD
}

def get_random_move() -> Move:
    return random.choice(list(MOVES_MAP.values()))()




# UR
# UL
# ML
# MR
# DL
# DR
# LU
# LD
# MU
# MD
# RU
# RD



