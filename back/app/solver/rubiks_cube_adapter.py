from typing import Sequence
import numpy as np
import random

import torch

from core.cube import Cube
from core.moves import get_move_by_name



class RubiksCubeAdapter:    
    # Use only the 12 basic moves for DeepCubeA (excluding M, E, S moves)
    LEGAL_MOVES = [
        "U", "U'", "U2",
        "D", "D'", "D2",
        "L", "L'", "L2",
        "R", "R'", "R2",
        "F", "F'", "F2",
        "B", "B'", "B2"
    ]
    
    # Inverse moves for efficient neighbor generation
    INVERSE_MOVES = {
        "U": "U'", "U'": "U", "U2": "U2",
        "D": "D'", "D'": "D", "D2": "D2",
        "L": "L'", "L'": "L", "L2": "L2",
        "R": "R'", "R'": "R", "R2": "R2",
        "F": "F'", "F'": "F", "F2": "F2",
        "B": "B'", "B'": "B", "B2": "B2"
    }
    
    
    def __init__(self):
        # 54 stickers × 6 colors = 324 dimensions for one-hot encoding
        self.input_dim = 54 * 6
        self.num_stickers = 54
        self.num_colors = 6
        
        
    def goal_state(self) -> np.ndarray:
        cube = Cube()
        return cube.state.copy()
    
    
    def is_goal(self, state: np.ndarray) -> bool:
        for face in state:
            ref_color = face[0, 0]
            if not np.all(face == ref_color):
                return False
        return True
    
    
    def random_scramble(self, state: np.ndarray, k: int) -> np.ndarray:
        # Create a copy to avoid modifying the original
        scrambled_state = state.copy()
        
        # Create a temporary Cube object for applying moves
        temp_cube = Cube()
        temp_cube.state = scrambled_state
        
        # Apply k random moves
        prev_move = None
        for _ in range(k):
            # Avoid applying the same face twice in a row (optimization)
            available_moves = self.LEGAL_MOVES.copy()
            if prev_move:
                # Remove moves of the same face
                face = prev_move[0]
                available_moves = [m for m in available_moves if m[0] != face]
            
            move_name = random.choice(available_moves)
            move = get_move_by_name(move_name)
            move.make_move_on_cube(temp_cube)
            prev_move = move_name
            
        return temp_cube.state.copy()
    
    
    def neighbors(self, state: np.ndarray) -> Sequence[np.ndarray]:
        neighbors = []
        
        for move_name in self.LEGAL_MOVES:
            # Create a temporary cube with the current state
            temp_cube = Cube()
            temp_cube.state = state.copy()
            
            # Apply the move
            move = get_move_by_name(move_name)
            move.make_move_on_cube(temp_cube)
            
            # Add the resulting state
            neighbors.append(temp_cube.state.copy())
            
        return neighbors
    
    
    def encode(self, state: np.ndarray) -> torch.Tensor:
        # Flatten the cube to get all 54 stickers
        flat_state = state.flatten()  # Shape: (54,)
        
        # Create one-hot encoding
        one_hot = torch.zeros(self.num_stickers * self.num_colors, dtype=torch.float32)
        
        for sticker_idx, color in enumerate(flat_state):
            # Calculate position in the one-hot vector
            # Each sticker gets 6 positions (one for each color)
            start_idx = sticker_idx * self.num_colors
            one_hot[start_idx + int(color)] = 1.0
            
        return one_hot
    
    
    def decode_action(self, action_idx: int) -> str:
        if 0 <= action_idx < len(self.LEGAL_MOVES):
            return self.LEGAL_MOVES[action_idx]
        else:
            raise ValueError(f"Invalid action index: {action_idx}")
    
    
    def get_action_index(self, move_name: str) -> int:
        try:
            return self.LEGAL_MOVES.index(move_name)
        except ValueError:
            raise ValueError(f"Invalid move name: {move_name}")
    
    
    def apply_move_to_state(self, state: np.ndarray, move_name: str) -> np.ndarray:
        temp_cube = Cube()
        temp_cube.state = state.copy()
        move = get_move_by_name(move_name)
        move.make_move_on_cube(temp_cube)
        return temp_cube.state.copy()


