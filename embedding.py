"""The token and position embedding stage of the forward pass."""

import torch


def embedding(self, x: torch.Tensor) -> torch.Tensor:
    """Return x0 [batch, context, model width] from one-hot x.

    Multiply x [batch, context, vocab] by W_E [vocab, model width],
    then add W_p [context, model width] at every batch item (broadcasting).
    """
    x0 = x @ self.W_E + self.W_p
    return x0
