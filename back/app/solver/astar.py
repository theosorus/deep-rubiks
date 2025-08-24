# back/app/solver/astar.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Set, Dict
import heapq
import time
import numpy as np
import torch

from solver.rubiks_cube_adapter import RubiksCubeAdapter
from solver.cost_to_go_net import CostToGoNet
from core.cube import Cube
from solver.config import AStarConfig


@dataclass
class SearchNode:
    state: np.ndarray
    parent: Optional[SearchNode]
    action: Optional[str]  # Move that led to this state
    g: float  # Cost from start
    h: float  # Heuristic (from neural network)
    f: float = field(init=False)  # Total cost (g + weight * h)
    
    def __post_init__(self):
        self.f = self.g + self.h
    
    def __lt__(self, other: SearchNode) -> bool:
        return self.f < other.f
    
    def __eq__(self, other: SearchNode) -> bool:
        return np.array_equal(self.state, other.state)
    
    def __hash__(self) -> int:
        return hash(self.state.tobytes())



@dataclass
class SearchResult:
    solved: bool
    solution_moves: List[str]
    solution_length: int
    nodes_expanded: int
    nodes_generated: int
    search_time: float
    final_state: np.ndarray


class AStarSolver:
    
    def __init__(self, net: CostToGoNet, adapter: RubiksCubeAdapter, device: str = 'cuda'):
        self.net = net
        self.adapter = adapter
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.net.to(self.device)
        self.net.eval()
    
    def _evaluate_heuristic(self, states: List[np.ndarray]) -> List[float]:
        with torch.no_grad():
            # Encode states
            encoded = [self.adapter.encode(state) for state in states]
            batch = torch.stack(encoded).to(self.device)
            
            # Get cost-to-go predictions
            costs = self.net(batch).squeeze(-1).cpu().numpy()
            
        return costs.tolist()
    
    def _evaluate_single_heuristic(self, state: np.ndarray) -> float:
        with torch.no_grad():
            encoded = self.adapter.encode(state).unsqueeze(0).to(self.device)
            cost = self.net(encoded).item()
        return cost
    
    def _reconstruct_path(self, node: SearchNode) -> List[str]:
        moves = []
        current = node
        while current.parent is not None:
            moves.append(current.action)
            current = current.parent
        return list(reversed(moves))
    
    def _state_to_key(self, state: np.ndarray) -> bytes:
        return state.tobytes()
    
    def solve(self, initial_state: np.ndarray, config: Optional[AStarConfig] = None) -> SearchResult:

        if config is None:
            config = AStarConfig()
        
        start_time = time.time()
        
        # Check if already solved
        if self.adapter.is_goal(initial_state):
            return SearchResult(
                solved=True,
                solution_moves=[],
                solution_length=0,
                nodes_expanded=0,
                nodes_generated=1,
                search_time=0.0,
                final_state=initial_state
            )
        
        # Initialize search
        initial_h = self._evaluate_single_heuristic(initial_state)
        start_node = SearchNode(
            state=initial_state,
            parent=None,
            action=None,
            g=0,
            h=initial_h
        )
        start_node.f = start_node.g + config.weight * start_node.h
        
        # Priority queue (min-heap)
        open_list = [start_node]
        open_dict: Dict[bytes, SearchNode] = {self._state_to_key(initial_state): start_node}
        
        # Closed set
        closed_set: Set[bytes] = set()
        
        # Statistics
        nodes_expanded = 0
        nodes_generated = 1
        
        if config.verbose:
            print(f"Starting A* search with weight={config.weight}")
            print(f"Initial heuristic: {initial_h:.2f}")
        
        while open_list and nodes_expanded < config.max_nodes:
            # Check timeout
            if time.time() - start_time > config.timeout:
                if config.verbose:
                    print(f"Search timeout after {config.timeout}s")
                break
            
            # Pop best node
            current = heapq.heappop(open_list)
            current_key = self._state_to_key(current.state)
            
            # Remove from open dict
            if current_key in open_dict:
                del open_dict[current_key]
            
            # Skip if already in closed set
            if current_key in closed_set:
                continue
            
            # Add to closed set
            closed_set.add(current_key)
            nodes_expanded += 1
            
            # Check if goal
            if self.adapter.is_goal(current.state):
                if config.verbose:
                    print(f"Solution found! Expanded {nodes_expanded} nodes")
                
                solution_moves = self._reconstruct_path(current)
                return SearchResult(
                    solved=True,
                    solution_moves=solution_moves,
                    solution_length=len(solution_moves),
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    search_time=time.time() - start_time,
                    final_state=current.state
                )
            
            # Generate neighbors
            neighbors_states = self.adapter.neighbors(current.state)
            neighbors_to_eval = []
            neighbors_data = []
            
            for i, neighbor_state in enumerate(neighbors_states):
                neighbor_key = self._state_to_key(neighbor_state)
                
                # Skip if in closed set
                if neighbor_key in closed_set:
                    continue
                
                # Prepare for batch evaluation
                move_name = self.adapter.LEGAL_MOVES[i]
                neighbors_to_eval.append(neighbor_state)
                neighbors_data.append((neighbor_state, neighbor_key, move_name))
            
            # Batch evaluate heuristics
            if neighbors_to_eval:
                heuristics = self._evaluate_heuristic(neighbors_to_eval)
                
                for (neighbor_state, neighbor_key, move_name), h in zip(neighbors_data, heuristics):
                    g = current.g + 1  # Each move costs 1
                    
                    # Check if already in open list with better cost
                    if neighbor_key in open_dict:
                        existing = open_dict[neighbor_key]
                        if g >= existing.g:
                            continue
                    
                    # Create new node
                    neighbor_node = SearchNode(
                        state=neighbor_state,
                        parent=current,
                        action=move_name,
                        g=g,
                        h=h
                    )
                    neighbor_node.f = g + config.weight * h
                    
                    # Add to open list
                    heapq.heappush(open_list, neighbor_node)
                    open_dict[neighbor_key] = neighbor_node
                    nodes_generated += 1
            
            # Progress reporting
            if config.verbose and nodes_expanded % 1000 == 0:
                print(f"Expanded: {nodes_expanded}, Open: {len(open_list)}, "
                      f"f={current.f:.2f}, g={current.g}, h={current.h:.2f}")
        
        # No solution found
        if config.verbose:
            print(f"No solution found. Expanded {nodes_expanded} nodes")
        
        return SearchResult(
            solved=False,
            solution_moves=[],
            solution_length=-1,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            search_time=time.time() - start_time,
            final_state=initial_state
        )


class BatchAStarSolver(AStarSolver):
    
    def solve_batch(self, initial_states: List[np.ndarray], config: Optional[AStarConfig] = None) -> List[SearchResult]:
        results = []
        for state in initial_states:
            result = self.solve(state, config)
            results.append(result)
        return results
    
    
    def solve_with_beam_search(self, initial_state: np.ndarray, beam_width: int = 100, 
                              max_depth: int = 30) -> SearchResult:
        start_time = time.time()
        
        if self.adapter.is_goal(initial_state):
            return SearchResult(
                solved=True,
                solution_moves=[],
                solution_length=0,
                nodes_expanded=0,
                nodes_generated=1,
                search_time=0.0,
                final_state=initial_state
            )
        
        # Initialize beam
        initial_h = self._evaluate_single_heuristic(initial_state)
        start_node = SearchNode(
            state=initial_state,
            parent=None,
            action=None,
            g=0,
            h=initial_h
        )
        
        beam = [start_node]
        nodes_expanded = 0
        nodes_generated = 1
        
        for depth in range(max_depth):
            next_beam = []
            
            # Expand all nodes in current beam
            for node in beam:
                nodes_expanded += 1
                
                # Check if goal
                if self.adapter.is_goal(node.state):
                    solution_moves = self._reconstruct_path(node)
                    return SearchResult(
                        solved=True,
                        solution_moves=solution_moves,
                        solution_length=len(solution_moves),
                        nodes_expanded=nodes_expanded,
                        nodes_generated=nodes_generated,
                        search_time=time.time() - start_time,
                        final_state=node.state
                    )
                
                # Generate neighbors
                neighbors_states = self.adapter.neighbors(node.state)
                
                # Batch evaluate
                if neighbors_states:
                    heuristics = self._evaluate_heuristic(neighbors_states)
                    
                    for i, (neighbor_state, h) in enumerate(zip(neighbors_states, heuristics)):
                        move_name = self.adapter.LEGAL_MOVES[i]
                        neighbor_node = SearchNode(
                            state=neighbor_state,
                            parent=node,
                            action=move_name,
                            g=node.g + 1,
                            h=h
                        )
                        next_beam.append(neighbor_node)
                        nodes_generated += 1
            
            # Keep only top beam_width nodes
            next_beam.sort(key=lambda n: n.f)
            beam = next_beam[:beam_width]
            
            if not beam:
                break
        
        return SearchResult(
            solved=False,
            solution_moves=[],
            solution_length=-1,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            search_time=time.time() - start_time,
            final_state=initial_state
        )


def solve_cube(cube: Cube, model_path: str, 
               config: Optional[AStarConfig] = None) -> SearchResult:
    from solver.utils import load_model
    
    print(f"Model path provided: {model_path}, loading model...")
    artifacts = load_model(model_path)

    # Create solver
    adapter = RubiksCubeAdapter()
    solver = AStarSolver(artifacts.net, adapter)
    
    # Solve
    result = solver.solve(cube.state, config)
    
    return result



   