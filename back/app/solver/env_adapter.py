from typing import Any, Protocol, Sequence
import torch

# ----------------------------
# Env Adapter (protocol)
# ----------------------------

class EnvAdapter(Protocol):
    """Adapter/protocol an environment must implement to be trainable via DAVI.

    State objects can be any Python object that your encode() method understands.
    """
    def goal_state(self) -> Any: ...
    def is_goal(self, state: Any) -> bool: ...
    def random_scramble(self, state: Any, k: int) -> Any: ...
    def neighbors(self, state: Any) -> Sequence[Any]:
        """Return the next reachable states for all legal actions from `state`. Order doesn't matter."""
        ...
    def encode(self, state: Any) -> torch.Tensor:
        """Return a 1D float tensor representing state (e.g., one-hot). Shape: [input_dim]."""
        ...