from solver.train import train_small_model,evaluate_solver,save_model
import torch





def main():
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




if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    if device == 'cpu':
        print("⚠️  GPU not available. Training will be slower on CPU.")

    main()