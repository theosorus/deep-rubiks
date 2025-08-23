
# davi_cost_to_go.py
# Deep Approximate Value Iteration (DAVI) for learning a cost-to-go function.
# Based on Agostinelli et al., 'Solving the Rubik's cube with deep reinforcement learning and search' (DeepCubeA).
# This file implements:
#   - A pluggable EnvAdapter interface for puzzles (Rubik's cube, sliding puzzles, etc.)
#   - A neural network j_theta(s) approximating the cost-to-go J(s)
#   - A DAVI training loop using one-step lookahead targets: J'(s) = min_a [ g(s, A(s,a)) + J_e(A(s,a)) ], with g=1.
#   - A simple example "MockEnv" to verify the code structure.
#
# Notes:
#   - To use with a real Rubik's cube implementation, implement an adapter that exposes the methods specified in EnvAdapter.
#   - For the Cube, your adapter should:
#       * return a solved/goal state
#       * generate k-step random scrambles (from the goal) with legal moves
#       * enumerate neighbors via all legal actions
#       * encode states as one-hot (e.g., 54 stickers x 6 colors -> 324 dims) as in the paper.
#

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple, Optional
import copy
import random
import time

import torch

from solver.env_adapter import EnvAdapter
from solver.cost_to_go_net import CostToGoNet
from solver.LineWorld import LineWorld



# ----------------------------
# Training (DAVI)
# ----------------------------

@dataclass
class DaviConfig:
    input_dim: int
    K: int = 30                    # max scrambles
    batch_size: int = 1024
    iterations: int = 10000
    lr: float = 1e-3
    check_every: int = 5000        # how often to check theta_e update criterion
    epsilon: float = 0.05          # loss threshold to refresh target
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_res_blocks: int = 4
    hidden1: int = 5000
    hidden2: int = 1000
    seed: Optional[int] = 42
    grad_clip_norm: Optional[float] = 1.0
    log_every: int = 10

@dataclass
class DaviArtifacts:
    net: CostToGoNet
    target_net: CostToGoNet
    history: List[Tuple[int, float]]  # (iteration, loss)
    cfg: DaviConfig

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def train_davi(env: EnvAdapter, cfg: DaviConfig) -> DaviArtifacts:
    if cfg.seed is not None:
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

        # Target network refresh heuristic from paper: when loss < epsilon at check interval
        if (it % cfg.check_every == 0) and (loss.item() < cfg.epsilon):
            target_net.load_state_dict(net.state_dict())

    dt = time.time() - t0
    print(f"Training finished in {dt:.1f}s. Last loss={loss.item():.4f}")
    return DaviArtifacts(net=net, target_net=target_net, history=history, cfg=cfg)



# ----------------------------
# Convenience wrappers
# ----------------------------

def run_sanity_check():
    env = LineWorld(n=8)
    cfg = DaviConfig(input_dim=8, K=6, batch_size=64, iterations=1000, check_every=1, log_every=1)
    artifacts = train_davi(env, cfg)
    return artifacts



if __name__ == "__main__":
    # Optional smoke test on the mock environment
    run_sanity_check()
