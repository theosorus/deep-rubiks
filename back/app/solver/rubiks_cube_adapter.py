# rubiks_cube_adapter.py
"""
Rubik's Cube adapter for DAVI (Deep Approximate Value Iteration).
This adapter implements the EnvAdapter protocol to enable training a neural network
to solve the Rubik's Cube using the DeepCubeA approach.

Based on: "Solving the Rubik's cube with deep reinforcement learning and search"
by Agostinelli et al. (2019)
"""

from typing import Any, List, Sequence
import numpy as np
import torch
import copy
import random
import os
import sys

# Add the app directory to path to import core modules
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
sys.path.insert(0, app_dir)

from core.cube import Cube
from core.moves import get_move_by_name
from solver.env_adapter import EnvAdapter


class RubiksCubeAdapter:
    """Adapter for Rubik's Cube to work with DAVI training."""
    
    # Use only the 12 basic moves for DeepCubeA (excluding M, E, S moves)
    # This matches the standard approach in the paper
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
        """Initialize the Rubik's Cube adapter."""
        # 54 stickers × 6 colors = 324 dimensions for one-hot encoding
        self.input_dim = 54 * 6
        self.num_stickers = 54
        self.num_colors = 6
        
    def goal_state(self) -> np.ndarray:
        """
        Return a solved Rubik's Cube state as numpy array.
        Using numpy arrays instead of Cube objects for better deepcopy compatibility.
        """
        cube = Cube()
        return cube.state.copy()
    
    def is_goal(self, state: np.ndarray) -> bool:
        """Check if the cube is in the solved state."""
        # Each face should have all stickers of the same color
        for face in state:
            ref_color = face[0, 0]
            if not np.all(face == ref_color):
                return False
        return True
    
    def random_scramble(self, state: np.ndarray, k: int) -> np.ndarray:
        """
        Generate a k-step scramble from the given state.
        
        Args:
            state: Starting cube state (typically the goal state)
            k: Number of random moves to apply
            
        Returns:
            Scrambled cube state as numpy array
        """
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
        """
        Return all states reachable by one legal move from the current state.
        
        Args:
            state: Current cube state as numpy array
            
        Returns:
            List of neighboring states (one for each legal move)
        """
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
        """
        Encode the cube state as a one-hot vector.
        
        The encoding follows the DeepCubeA paper:
        - 54 stickers total (9 per face × 6 faces)
        - Each sticker is one-hot encoded over 6 possible colors
        - Result: 54 × 6 = 324 dimensional vector
        
        Args:
            state: Cube state to encode (numpy array of shape (6, 3, 3))
            
        Returns:
            1D tensor of shape [324] with one-hot encoding
        """
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
        """
        Convert action index to move name.
        
        Args:
            action_idx: Index of the action (0-17 for 18 legal moves)
            
        Returns:
            Move name string
        """
        if 0 <= action_idx < len(self.LEGAL_MOVES):
            return self.LEGAL_MOVES[action_idx]
        else:
            raise ValueError(f"Invalid action index: {action_idx}")
    
    def get_action_index(self, move_name: str) -> int:
        """
        Convert move name to action index.
        
        Args:
            move_name: Name of the move
            
        Returns:
            Index of the action
        """
        try:
            return self.LEGAL_MOVES.index(move_name)
        except ValueError:
            raise ValueError(f"Invalid move name: {move_name}")
    
    def apply_move_to_state(self, state: np.ndarray, move_name: str) -> np.ndarray:
        """
        Apply a single move to a state and return the new state.
        
        Args:
            state: Current cube state
            move_name: Name of the move to apply
            
        Returns:
            New state after applying the move
        """
        temp_cube = Cube()
        temp_cube.state = state.copy()
        move = get_move_by_name(move_name)
        move.make_move_on_cube(temp_cube)
        return temp_cube.state.copy()


# ----------------------------
# Helper functions for training
# ----------------------------

def create_training_config(
    iterations: int = 100000,
    batch_size: int = 10000,
    K: int = 30,
    lr: float = 1e-4,
    check_every: int = 1000,
    epsilon: float = 0.01,
    num_res_blocks: int = 4,
    hidden1: int = 5000,
    hidden2: int = 1000,
    device: str = None,
    log_every: int = 100
):
    """
    Create a DAVI configuration for Rubik's Cube training.
    
    These hyperparameters are based on the DeepCubeA paper:
    - Large batch size (10k) for stable training
    - K=30 for maximum scramble depth during training
    - Low learning rate (1e-4) for stability
    - 4 residual blocks as in the paper
    - Hidden layers: 5000 -> 1000 neurons
    
    Args:
        iterations: Number of training iterations
        batch_size: Batch size for training
        K: Maximum scramble depth
        lr: Learning rate
        check_every: Frequency of target network updates
        epsilon: Loss threshold for target network refresh
        num_res_blocks: Number of residual blocks in the network
        hidden1: Size of first hidden layer
        hidden2: Size of second hidden layer
        device: Device to use ('cuda' or 'cpu', auto-detect if None)
        log_every: Frequency of logging
        
    Returns:
        DaviConfig object
    """
    from solver.davi import DaviConfig
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    adapter = RubiksCubeAdapter()
    
    return DaviConfig(
        input_dim=adapter.input_dim,
        K=K,
        batch_size=batch_size,
        iterations=iterations,
        lr=lr,
        check_every=check_every,
        epsilon=epsilon,
        device=device,
        num_res_blocks=num_res_blocks,
        hidden1=hidden1,
        hidden2=hidden2,
        seed=42,
        grad_clip_norm=1.0,
        log_every=log_every
    )


def train_rubiks_cube_solver(config=None, verbose=True):
    """
    Train a neural network to solve the Rubik's Cube using DAVI.
    
    Args:
        config: DaviConfig object (uses defaults if None)
        verbose: Whether to print training progress
        
    Returns:
        DaviArtifacts containing trained networks and training history
    """
    from solver.davi import train_davi
    
    # Create adapter
    adapter = RubiksCubeAdapter()
    
    # Use default config if not provided
    if config is None:
        config = create_training_config()
    
    if verbose:
        print(f"Starting Rubik's Cube DAVI training...")
        print(f"Device: {config.device}")
        print(f"Iterations: {config.iterations}")
        print(f"Batch size: {config.batch_size}")
        print(f"Max scramble depth: {config.K}")
        print(f"Learning rate: {config.lr}")
        print(f"Network: {config.hidden1} -> {config.hidden2} -> {config.num_res_blocks} res blocks")
        print("-" * 50)
    
    # Train the model
    artifacts = train_davi(adapter, config)
    
    if verbose:
        print("-" * 50)
        print(f"Training complete!")
        if artifacts.history:
            print(f"Initial loss: {artifacts.history[0][1]:.4f}")
            print(f"Final loss: {artifacts.history[-1][1]:.4f}")
    
    return artifacts


def evaluate_solver(artifacts, num_tests: int = 100, max_scramble: int = 10, verbose=True):
    """
    Evaluate the trained solver on random scrambles.
    
    Args:
        artifacts: DaviArtifacts from training
        num_tests: Number of test scrambles
        max_scramble: Maximum scramble depth for testing
        verbose: Whether to print results
        
    Returns:
        Dictionary with evaluation metrics
    """
    adapter = RubiksCubeAdapter()
    net = artifacts.net
    net.eval()
    
    results = {
        'scramble_depth': [],
        'predicted_cost': [],
    }
    
    with torch.no_grad():
        for depth in range(1, max_scramble + 1):
            for _ in range(num_tests // max_scramble):
                # Create a scrambled cube
                goal = adapter.goal_state()
                scrambled = adapter.random_scramble(goal, depth)
                
                # Get predicted cost-to-go
                encoded = adapter.encode(scrambled).unsqueeze(0).to(artifacts.cfg.device)
                cost = net(encoded).item()
                
                results['scramble_depth'].append(depth)
                results['predicted_cost'].append(cost)
    
    # Calculate statistics
    avg_cost_by_depth = {}
    std_cost_by_depth = {}
    for depth in range(1, max_scramble + 1):
        costs = [c for d, c in zip(results['scramble_depth'], results['predicted_cost']) if d == depth]
        if costs:
            avg_cost_by_depth[depth] = np.mean(costs)
            std_cost_by_depth[depth] = np.std(costs)
    
    if verbose:
        print("\nEvaluation Results:")
        print("Scramble Depth | Avg Predicted Cost | Std Dev")
        print("-" * 50)
        for depth in range(1, max_scramble + 1):
            if depth in avg_cost_by_depth:
                print(f"{depth:14d} | {avg_cost_by_depth[depth]:18.2f} | {std_cost_by_depth[depth]:7.2f}")
    
    return {
        'avg_cost_by_depth': avg_cost_by_depth,
        'std_cost_by_depth': std_cost_by_depth,
        'raw_results': results
    }


def save_model(artifacts, filepath: str):
    """
    Save the trained model to disk.
    
    Args:
        artifacts: DaviArtifacts from training
        filepath: Path to save the model
    """
    torch.save({
        'model_state_dict': artifacts.net.state_dict(),
        'target_model_state_dict': artifacts.target_net.state_dict(),
        'config': artifacts.cfg,
        'history': artifacts.history
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str, device=None):
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        device: Device to load the model to
        
    Returns:
        Loaded artifacts
    """
    from solver.davi import DaviArtifacts
    from solver.cost_to_go_net import CostToGoNet
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(filepath, map_location=device)
    
    cfg = checkpoint['config']
    cfg.device = device
    
    net = CostToGoNet(
        cfg.input_dim,
        hidden1=cfg.hidden1,
        hidden2=cfg.hidden2,
        num_res_blocks=cfg.num_res_blocks
    ).to(device)
    
    target_net = CostToGoNet(
        cfg.input_dim,
        hidden1=cfg.hidden1,
        hidden2=cfg.hidden2,
        num_res_blocks=cfg.num_res_blocks
    ).to(device)
    
    net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['target_model_state_dict'])
    
    artifacts = DaviArtifacts(
        net=net,
        target_net=target_net,
        history=checkpoint['history'],
        cfg=cfg
    )
    
    print(f"Model loaded from {filepath}")
    return artifacts