

def loss_fn( y, z ) :
    """
    Compute mean cross-entropy for the answer token (last position) only.
    y contains target probabilities and z contains model logits; both have
    shape (batch, d_context, d_vocab). Return one scalar loss.
    """
    y = y[:,-1,:]
    probs = torch.log(torch.softmax(z[:,-1,:],dim = 1))
    loss = -(y @ probs.T).sum()/(y.shape[0] * y.shape[1])

    return loss
