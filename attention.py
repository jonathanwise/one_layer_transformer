"""The four-head attention stage of the forward pass."""

import torch



def attention(self, x0: torch.Tensor) -> torch.Tensor:
    """Return x1 of shape [batch, context, model width] from embedded input 
    x0 of same shape.

    For each of the four heads h, compute Q[h]=x0*W_Q[h], K[h]=x0*W_K[h],
    V[h]=x0*W_V[h], score matrix S=Q*K^t, and attention weights A=softmax(S).
    Output of each head is A*V*W_O[h], sum the
    projected head outputs, and add the residual x0.
    """
    x = x0.unsqueeze(-3)                      # [..., 1, context, width] -> broadcasts over heads
    Q = x @ self.W_Q                          # [..., heads, context, head width]
    K = x @ self.W_K
    V = x @ self.W_V
    S = Q @ K.transpose(-2, -1)               # [..., heads, context, context]
    A = torch.softmax(S, dim=-1)
    head_out = A @ V @ self.W_O               # [..., heads, context, width]
    x1 = head_out.sum(dim=-3) + x0            # sum over heads, add residual
    return x1


if __name__ == "__main__":
    from model import *
    from config import *

    c = config()
    m = model(c)
    x = torch.randn(3, c.d_model, dtype=torch.float32)
    print(attention(m, x).shape)              # should match x.shape