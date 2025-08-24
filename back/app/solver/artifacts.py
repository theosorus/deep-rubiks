from dataclasses import dataclass
from solver.cost_to_go_net import CostToGoNet   
from solver.config import DaviConfig

from typing import List, Tuple

@dataclass
class DaviArtifacts:
    net: CostToGoNet
    target_net: CostToGoNet
    history: List[Tuple[int, float]]  # (iteration, loss)
    cfg: DaviConfig
