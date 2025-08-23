# models/cube.py
import numpy as np
import random
from core.colors import COLORS_TO_INT,INT_TO_COLORS
from core.face import DEFAULT_FACE_COLORS,DEFAULT_FACE_ORDER
from core.moves import get_random_moves


class Cube:
    def __init__(self,initial_state:np.ndarray = None) -> None:
        if initial_state:
            assert initial_state.shape == (6, 3, 3), "Initial state must be of shape (6, 3, 3)"
            self.state = initial_state
        else:
            self.state = self._init_cube()

    def _init_cube(self) -> np.ndarray:
        state = np.empty((6,3,3), dtype=object)
        for idx,face in enumerate(DEFAULT_FACE_ORDER):
            color = DEFAULT_FACE_COLORS[face]
            color_int = COLORS_TO_INT[color]
            state[idx] = np.full((3,3), color_int, dtype=object)
        return state
        
    def shuffle_cube(self, min_move: int = 100, max_move: int = 500) -> None:
        moves = get_random_moves(random.randint(min_move, max_move))
        for m in moves:
            m.make_move_on_cube(self)

    def get_face(self, k: str) -> np.ndarray:
        index = self.get_face_index(k)
        return self.state[index]
    
    def get_face_index(self, k: str) -> int:
        index = DEFAULT_FACE_ORDER.index(k)
        if index < 0 or index >= len(self.state):
            raise ValueError(f"Invalid face key: {k}. Must be one of {DEFAULT_FACE_ORDER}.")
        return index
    
    def reset_cube(self) -> None:
        self.state = self._init_cube()
    
    @property
    def flatten_cube(self) -> np.ndarray :
        return self.state.flatten()

    def set_face(self, k: str, arr: np.ndarray) -> None:
        if arr.shape != (3, 3):
            raise ValueError(f"Face array must be of shape (3, 3), got {arr.shape}.")
        index = self.get_face_index(k)
        self.state[index] = arr

    def __dict__(self):
        return {
            "state": self.state.tolist(),
            "flatten_state": self.flatten_cube.tolist()  ,
            "colors" : INT_TO_COLORS,  
            "face_order": DEFAULT_FACE_ORDER
        }

    def __str__(self) -> str:
        s = ["----------"]
        for k in DEFAULT_FACE_ORDER:
            s.append(f"{k} face:")
            for row in self.state[self.get_face_index(k)]:
                s.append("[ " + "  ".join(f"{str(INT_TO_COLORS[v]):7}" for v in row) + " ]")
        s.append("----------")
        return "\n".join(s)
    
    def __eq__(self, value):
        if not isinstance(value, Cube):
            return False
        return np.array_equal(self.state, value.state)
    
    def is_solved(self) -> bool:
        for face in self.state:
            ref = face[0, 0]
            if not np.all(face == ref):
                return False
        return True