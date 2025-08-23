import random

import torch 

# ----------------------------
# Example adapter (mock)
# ----------------------------
# A tiny reversible environment with 8 states in a line: 0 -- 1 -- 2 -- ... -- 7
# Goal is state 0. Actions: left/right. This is only for sanity-checking the training pipeline.

class LineWorld:
    def __init__(self, n: int = 8):
        self.n = n

    def goal_state(self) -> int:
        return 0

    def is_goal(self, state: int) -> bool:
        return state == 0

    def random_scramble(self, state: int, k: int) -> int:
        s = state
        for _ in range(k):
            if random.random() < 0.5:  # move right if possible
                s = min(self.n - 1, s + 1)
            else:                      # move left if possible
                s = max(0, s - 1)
        return s

    def neighbors(self, state: int):
        nbrs = []
        if state > 0:
            nbrs.append(state - 1)
        if state < self.n - 1:
            nbrs.append(state + 1)
        return nbrs

    def encode(self, state: int) -> torch.Tensor:
        # One-hot over n states
        v = torch.zeros(self.n, dtype=torch.float32)
        v[state] = 1.0
        return v