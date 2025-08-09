import numpy as np
from models.colors import Colors
from models.moves import get_random_moves

import random


class Cube:
    def __init__(self):
        self.state = self._init_cube() # [6,3,3]

    def _init_cube(self) -> np.ndarray:
        return np.array([np.full((3, 3), c.value) for c in Colors])

    def shuffle_cube(self,min_move:int = 100,max_move: int = 500) -> None:
        moves = get_random_moves(random.randint(min_move, max_move))
        for move in moves:
            move.make_move_on_cube(self)
        

    @property
    def flatten_cube(self) -> np.ndarray :
        return self.state.flatten()
    
    def __dict__(self):
        return {
            "state": self.state.tolist(),
            "flatten_state": self.flatten_cube.tolist(),        
            "colors" : {color : color.name for color in Colors}    
        }


