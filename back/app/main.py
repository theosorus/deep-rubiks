from solver.train import train_small_model,evaluate_solver,save_model
import torch

from solver.utils import get_device

from core.cube import Cube
from solver.astar import AStarConfig, solve_cube
from core.moves import apply_moves




def main_davi():
    """Main function to run tests and training."""
    
    
    # Train a small model for testing
    print("\n" + "=" * 60)
    print("Phase 1: Quick Training Test")
    print("=" * 60)
    
    artifacts = train_small_model()
    save_model(artifacts, "output/rubiks_cube_solver_small.pth")
    
    # Evaluate the small model
    print("\n" + "=" * 60)
    print("Evaluating Small Model...")
    print("=" * 60)
    
    eval_results = evaluate_solver(
        artifacts,
        num_tests=50,
        max_scramble=5,
        verbose=True
    )
    
    # Ask if user wants to train full model
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    print("\n✅ Small model training successful!")
    
def main_astar():
    cube = Cube()
    test_moves = ["R", "U", "R'", "U'"]  # Simple scramble
    apply_moves(cube, test_moves)
    
    print("Scrambled cube:")
    print(f"Applied moves: {test_moves}")
    print(f"Is solved: {cube.is_solved()}")
    
    # Solve it
    print("\nSolving...")
    config = AStarConfig(weight=1.5, verbose=True, max_nodes=10000)
    result = solve_cube(cube, model_path="output/rubiks_cube_solver_small.pth",config=config)
    
    if result.solved:
        print(f"\nSolution found!")
        print(f"Moves: {result.solution_moves}")
        print(f"Length: {result.solution_length}")
        print(f"Nodes expanded: {result.nodes_expanded}")
        print(f"Time: {result.search_time:.2f}s")
        
        # Verify solution
        cube_verify = Cube()
        apply_moves(cube_verify, test_moves)
        apply_moves(cube_verify, result.solution_moves)
        print(f"Verification: Cube solved = {cube_verify.is_solved()}")
    else:
        print("No solution found within limits")




if __name__ == "__main__":
    
    device = get_device()
    
    print(f"🖥️  Using device: {device}")
    if device == 'cpu':
        print("⚠️  GPU not available. Training will be slower on CPU.")

    # main_davi()
    main_astar()
    