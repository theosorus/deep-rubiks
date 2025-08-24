from dataclasses import dataclass
import torch
from typing import Optional

from solver.utils import get_device

@dataclass
class DaviConfig:
    input_dim: int
    K: int = 30                    # max scrambles
    batch_size: int = 1024
    iterations: int = 10000
    lr: float = 1e-3
    check_every: int = 5000        # how often to check theta_e update criterion
    epsilon: float = 0.05          # loss threshold to refresh target
    device: torch.device = get_device()
    num_res_blocks: int = 4
    hidden1: int = 5000
    hidden2: int = 1000
    seed: Optional[int] = 42
    grad_clip_norm: Optional[float] = 1.0
    log_every: int = 10
    
    
@dataclass  
class AStarConfig:
    """Configuration for A* search."""
    weight: float = 1.0  # Weight for weighted A* (1.0 = standard A*)
    max_nodes: int = 100000  # Maximum nodes to expand
    batch_size: int = 512  # Batch size for neural network evaluation
    timeout: float = 60.0  # Maximum search time in seconds
    verbose: bool = False
