"""The token and position embedding stage of the forward pass."""

import torch


def embedding(model, x: torch.Tensor) -> torch.Tensor:
    """Return x0 [batch, context, model width] from one-hot x.

    Multiply x [batch, context, vocab] by W_E [vocab, model width],
    then add W_p [context, model width] at every batch item (broadcasting).
    """
    logits = x @ model.W_E + model.W_p
    return logits
