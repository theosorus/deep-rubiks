# main_solveur.py
"""
Main script for training and testing the Rubik's Cube DAVI solver.
This implements the DeepCubeA approach for solving the Rubik's Cube.
"""

from solver.rubiks_cube_adapter import (
    RubiksCubeAdapter,
    create_training_config,
    train_rubiks_cube_solver,
    evaluate_solver,
    save_model,
    load_model
)
from solver.davi import DaviConfig, train_davi
import torch


def test_adapter():
    """Test basic functionality of the Rubik's Cube adapter."""
    print("=" * 60)
    print("Testing Rubik's Cube Adapter...")
    print("=" * 60)
    
    adapter = RubiksCubeAdapter()
    
    # Test basic properties
    print(f"\n✓ Input dimension: {adapter.input_dim} (54 stickers × 6 colors)")
    print(f"✓ Legal moves: {len(adapter.LEGAL_MOVES)} moves")
    
    # Test goal state
    goal = adapter.goal_state()
    is_solved = adapter.is_goal(goal)
    print(f"\n✓ Goal state created")
    print(f"✓ Goal state is solved: {is_solved}")
    assert is_solved, "Goal state should be solved!"
    
    # Test scrambling
    print(f"\n✓ Testing scrambling...")
    for k in [1, 5, 10, 20]:
        scrambled = adapter.random_scramble(goal, k=k)
        is_scrambled_solved = adapter.is_goal(scrambled)
        print(f"  - {k:2d}-move scramble is solved: {is_scrambled_solved}")
    
    # Test neighbors
    neighbors = adapter.neighbors(goal)
    print(f"\n✓ Number of neighbors from goal: {len(neighbors)}")
    print(f"  (Should be {len(adapter.LEGAL_MOVES)} for all legal moves)")
    
    # Test encoding
    encoding = adapter.encode(goal)
    print(f"\n✓ Encoding shape: {encoding.shape}")
    print(f"✓ Encoding sum: {encoding.sum().item()} (should be 54 for one-hot)")
    print(f"✓ Non-zero elements: {(encoding > 0).sum().item()} (should be 54)")
    
    # Verify encoding is valid one-hot
    assert encoding.shape[0] == 324, "Encoding should be 324-dimensional"
    assert encoding.sum().item() == 54, "Should have exactly 54 ones"
    assert ((encoding == 0) | (encoding == 1)).all(), "Should be binary"
    
    print("\n✅ All adapter tests passed!")
    return True


def train_small_model():
    """Train a small model for testing purposes."""
    print("\n" + "=" * 60)
    print("Training Small Test Model...")
    print("=" * 60)
    
    # Create a small configuration for quick testing
    config = create_training_config(
        iterations=500,      # Small number for testing
        batch_size=128,      # Smaller batch for faster iteration
        K=10,               # Smaller max scramble
        check_every=100,    # Check more frequently
        log_every=50,       # Log more frequently
        hidden1=256,        # Smaller network
        hidden2=128,        # Smaller network
        num_res_blocks=2    # Fewer residual blocks
    )
    
    print(f"\nTraining configuration:")
    print(f"  - Iterations: {config.iterations}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Max scramble: {config.K}")
    print(f"  - Network: {config.hidden1} -> {config.hidden2} -> {config.num_res_blocks} res blocks")
    
    # Train the model
    artifacts = train_rubiks_cube_solver(config, verbose=True)
    
    # Show training progress
    if artifacts.history:
        print(f"\n📊 Training Progress:")
        for i, (iteration, loss) in enumerate(artifacts.history[-5:]):
            print(f"  Iteration {iteration:4d}: Loss = {loss:.6f}")
    
    return artifacts


def train_full_model():
    """Train a full-scale model (as in the paper)."""
    print("\n" + "=" * 60)
    print("Training Full-Scale DeepCubeA Model...")
    print("=" * 60)
    
    # Use the paper's hyperparameters
    config = create_training_config(
        iterations=100000,   # Full training
        batch_size=10000,    # Large batch as in paper
        K=30,               # Max scramble depth
        lr=1e-4,            # Learning rate from paper
        check_every=1000,   # Target network update frequency
        epsilon=0.01,       # Target network update threshold
        hidden1=5000,       # Large first hidden layer
        hidden2=1000,       # Large second hidden layer
        num_res_blocks=4,   # 4 residual blocks as in paper
        log_every=1000      # Log less frequently
    )
    
    print(f"\nFull training configuration (DeepCubeA paper settings):")
    print(f"  - Iterations: {config.iterations:,}")
    print(f"  - Batch size: {config.batch_size:,}")
    print(f"  - Max scramble: {config.K}")
    print(f"  - Learning rate: {config.lr}")
    print(f"  - Network: {config.hidden1:,} -> {config.hidden2:,} -> {config.num_res_blocks} res blocks")
    print(f"\n⚠️  This will take a long time to train!")
    
    # Train the model
    artifacts = train_rubiks_cube_solver(config, verbose=True)
    
    # Save the trained model
    save_model(artifacts, "rubiks_cube_solver.pth")
    
    return artifacts


def main():
    """Main function to run tests and training."""
    
    # Test the adapter first
    test_success = test_adapter()
    
    if not test_success:
        print("\n❌ Adapter tests failed. Please fix the issues before training.")
        return
    
    # Train a small model for testing
    print("\n" + "=" * 60)
    print("Phase 1: Quick Training Test")
    print("=" * 60)
    
    artifacts = train_small_model()
    
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
    print("\nThe small model was trained with reduced parameters for testing.")
    print("For a production-ready solver, you should train the full model with:")
    print("  - 100,000+ iterations")
    print("  - Batch size of 10,000")
    print("  - Full network architecture (5000->1000->4 res blocks)")
    print("\nTo train the full model, uncomment the last line in main()")
    
    # Uncomment this line to train the full model:
    # artifacts_full = train_full_model()


if __name__ == "__main__":
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    if device == 'cpu':
        print("⚠️  GPU not available. Training will be slower on CPU.")
    
    # Run the main program
    main()