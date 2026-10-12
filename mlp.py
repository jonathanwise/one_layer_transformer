"""The MLP stage of the forward pass."""

import torch


def mlp(self, x1: torch.Tensor) -> torch.Tensor:
    """Return x2 [batch, context, model width] from attention output x1.

    Compute the hidden activation as ReLU(x1 * W_i + b_i),
    then return hidden * W_o + b_o.
    """
    mlp = torch.relu(x1 @ self.W_i + self.b_i)
    x2 = mlp @ self.W_o + self.b_o
    return x2
