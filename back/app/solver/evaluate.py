import torch
import numpy as np

from solver.rubiks_cube_adapter import RubiksCubeAdapter


def evaluate_solver(artifacts, num_tests: int = 100, max_scramble: int = 10, verbose=True):
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