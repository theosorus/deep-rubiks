import numpy as np
import torch
from solver.rubiks_cube_adapter import RubiksCubeAdapter



def train_small_model():
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
    
    # Train the model
    artifacts = train_rubiks_cube_solver(config, verbose=True)
    
    # Show training progress
    if artifacts.history:
        print(f"\n📊 Training Progress:")
        for i, (iteration, loss) in enumerate(artifacts.history[-5:]):
            print(f"  Iteration {iteration:4d}: Loss = {loss:.6f}")
    
    return artifacts



def train_full_model():
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


def train_rubiks_cube_solver(config=None, verbose=True):
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


def save_model(artifacts, filepath: str):
    torch.save({
        'model_state_dict': artifacts.net.state_dict(),
        'target_model_state_dict': artifacts.target_net.state_dict(),
        'config': artifacts.cfg,
        'history': artifacts.history
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str, device=None):
    from solver.davi import DaviArtifacts
    from solver.cost_to_go_net import CostToGoNet
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(filepath, map_location=device,weights_only=False)
    
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