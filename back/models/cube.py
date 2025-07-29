import numpy as np
from models.colors import Colors


class Cube:
    def __init__(self):
        self.state = self._init_cube() # [6,3,3]


        print(self.state.shape)
        print(self.state)
        print(self.flatten_cube().shape)

    def _init_cube(self) -> np.ndarray:
        return np.array([np.full((3, 3), c.value) for c in Colors])

    def suffle_cube(self,min_move:int = 100,max_move: int = 500) -> None:
        pass

    @property
    def flatten_cube(self) -> np.ndarray :
        return self.state.flatten


