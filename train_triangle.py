"""Train the one-layer transformer with a triangle-wave MLP activation.

The MLP's ReLU is replaced by 2 * |(x mod P) - P/2|; everything else is the
repository's code unchanged.

    python train_triangle.py --period 2 --steps 3000
    python train_triangle.py --period 2 --save triangle_P2.pt    # also save the weights
"""

import argparse
import dataclasses

import torch

from attention import attention
from config import default_config
from data import generate_data_tensor
from embedding import embedding
from loss import loss_fn
from model import model
from unembedding import unembedding


def triangle_wave(period):
    "Continuous periodic activation 2 * |(x mod P) - P/2|, ranging over [0, P]."
    return lambda x: 2 * torch.abs(torch.remainder(x, period) - period / 2)


class triangle_model(model):
    "The repository's transformer with the MLP's ReLU replaced by a triangle wave."

    def __init__(self, conf, period):
        super().__init__(conf)
        self.period = period

    def forward(self, x):
        x1 = attention(self, embedding(self, x))
        hidden = triangle_wave(self.period)(x1 @ self.W_i + self.b_i)
        return unembedding(self, hidden @ self.W_o + self.b_o)


def make_split(conf, device, train_frac=0.8, split_seed=0):
    "Shuffle all N**2 equations with a fixed seed and split them."
    torch.manual_seed(split_seed)
    X, Y = generate_data_tensor(conf)
    n_train = round(train_frac * len(X))
    return tuple(t.to(device) for t in (X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]))


@torch.no_grad()
def evaluate(net, X, Y):
    "Answer-token loss and accuracy."
    logits = net(X)
    accuracy = (logits[:, -1].argmax(-1) == Y[:, -1].argmax(-1)).float().mean().item()
    return loss_fn(Y, logits).item(), accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--period", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save", metavar="PATH", help="save the trained weights here (default: don't)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf = dataclasses.replace(default_config, logdir=f"runs/triangle_P{args.period:g}")
    X_train, Y_train, X_test, Y_test = make_split(conf, device)

    torch.manual_seed(args.seed)
    net = triangle_model(conf, args.period).to(device)

    for step in range(args.steps + 1):
        if step % args.log_every == 0 or step == args.steps:
            train_loss, train_acc = evaluate(net, X_train, Y_train)
            test_loss, test_acc = evaluate(net, X_test, Y_test)
            print(f"step {step:5d}  train loss {train_loss:.4f} acc {train_acc:.3f}   "
                  f"test loss {test_loss:.4f} acc {test_acc:.3f}")
        if step < args.steps:
            net.train(X_train, Y_train)

    if args.save:
        torch.save({name: tensor.cpu() for name, tensor in net.state_dict().items()}, args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
