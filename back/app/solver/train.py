import numpy as np
import torch
from solver.rubiks_cube_adapter import RubiksCubeAdapter
import time
import random
from typing import Any, List, Sequence, Tuple
import copy

from solver.env_adapter import EnvAdapter
from solver.utils import get_device, set_seed, save_model
from solver.config import DaviConfig
from solver.artifacts import DaviArtifacts
from solver.cost_to_go_net import CostToGoNet



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
        log_every=1,       # Log more frequently
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
        iterations=10000,   # Full training
        batch_size=1000,    # Large batch as in paper
        K=30,               # Max scramble depth
        lr=1e-4,            # Learning rate from paper
        check_every=1000,   # Target network update frequency
        epsilon=0.01,       # Target network update threshold
        hidden1=5000,       # Large first hidden layer
        hidden2=1000,       # Large second hidden layer
        num_res_blocks=4,   # 4 residual blocks as in paper
        log_every=1     # Log less frequently
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



def train_davi(env: EnvAdapter, cfg: DaviConfig) -> DaviArtifacts:
    if cfg.seed:
        set_seed(cfg.seed)

    net = CostToGoNet(cfg.input_dim, hidden1=cfg.hidden1, hidden2=cfg.hidden2, num_res_blocks=cfg.num_res_blocks).to(cfg.device)
    target_net = copy.deepcopy(net).eval().to(cfg.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    history: List[Tuple[int, float]] = []

    goal = env.goal_state()
    net.train()
    t0 = time.time()
    for it in range(1, cfg.iterations + 1):
        # Sample batch by scrambling from goal with k ~ Uniform(1..K)
        batch_states = []
        for _ in range(cfg.batch_size):
            k = random.randint(1, cfg.K)
            s = env.random_scramble(goal, k)
            batch_states.append(s)

        X = _encode_batch(env, batch_states, cfg.device)            # [B, D]
        y = _targets_one_step_lookahead(env, target_net, batch_states, cfg.device)  # [B]

        pred = net(X).squeeze(-1)  # [B]
        loss = torch.nn.functional.mse_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
        optimizer.step()

        if it % cfg.log_every == 0 or it == 1:
            history.append((it, float(loss.item())))
            # Add real-time logging here
            elapsed = time.time() - t0
            print(f"Iteration {it}/{cfg.iterations} [{it/cfg.iterations*100:.1f}%] - Loss: {loss.item():.6f} - Time: {elapsed:.1f}s")

        # Target network refresh heuristic from paper: when loss < epsilon at check interval
        if (it % cfg.check_every == 0) and (loss.item() < cfg.epsilon):
            target_net.load_state_dict(net.state_dict())
            print(f"✓ Target network updated at iteration {it} (loss = {loss.item():.6f})")

    dt = time.time() - t0
    print(f"Training finished in {dt:.1f}s. Last loss={loss.item():.4f}")
    return DaviArtifacts(net=net, target_net=target_net, history=history, cfg=cfg)


def _encode_batch(env: EnvAdapter, states: Sequence[Any], device: str) -> torch.Tensor:
    xs = [env.encode(s) for s in states]
    X = torch.stack(xs, dim=0).to(device)
    return X


def _targets_one_step_lookahead(env: EnvAdapter, target_net: CostToGoNet, states: Sequence[Any], device: str) -> torch.Tensor:
    """Compute y_i = min_a (1 + J_e(A(s_i, a))) ; and y=0 for goal states."""
    y_list: List[torch.Tensor] = []
    with torch.no_grad():
        for s in states:
            if env.is_goal(s):
                y_list.append(torch.tensor(0.0, device=device))
                continue
            nbrs = env.neighbors(s)
            if len(nbrs) == 0:
                # Dead-end: cost is 0 by convention (or could be inf). We'll use a large number to discourage it.
                y_list.append(torch.tensor(0.0, device=device))
                continue
            enc = _encode_batch(env, nbrs, device)  # [A, D]
            costs = target_net(enc).squeeze(-1)      # [A]
            # Each step cost is 1
            y = (1.0 + costs.min())
            y_list.append(y)
    return torch.stack(y_list, dim=0)  # [B]


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