"""Train with periodic weight snapshots and denser sampling after loss spikes."""

import math
from pathlib import Path

import torch

from loss import loss_fn


def train_with_snapshots(
    snapshot_interval,
    epochs,
    slingshot_snapshot_interval,
    slingshot_snapshot_count,
    threshold,
    in_training,
    out_training,
    trainer,
    in_testing,
    out_testing,
    prefix="weights",
):
    """Run full-batch epochs using an existing ``training.trainer`` instance.

    Call ``trainer.test`` before ``trainer.train`` each epoch so their logged
    losses share the same epoch index. Since trainer.train does not return its
    loss, measure full-batch training loss in an extra forward pass before the
    update. Compare successive measurements within this call to detect spikes.

    Save post-update model state_dicts to ``{prefix}/{trainer.epoch}.pt``;
    filenames count completed epochs (the first is 1 for a fresh trainer).
    Regular snapshots follow absolute multiples of snapshot_interval. A spike
    saves immediately, counting as the first of slingshot_snapshot_count dense
    snapshots. Subsequent dense snapshots are spaced by
    slingshot_snapshot_interval epochs. Another spike restarts that window.
    Regular snapshots continue and do not consume the dense snapshot count
    unless they coincide with a scheduled dense snapshot.

    Return a list of dictionaries containing epoch, training_loss, spike, and
    snapshot (a path string, or None). Test loss is recorded by trainer.test in
    TensorBoard. Only weights are saved, without optimizer state. Existing
    snapshot files at the same paths are overwritten.

    Starting values: snapshot_interval=100, epochs=10_000,
    slingshot_snapshot_interval=5, slingshot_snapshot_count=100, threshold=2.0.
    """
    for name, value in (
        ("snapshot_interval", snapshot_interval),
        ("slingshot_snapshot_interval", slingshot_snapshot_interval),
        ("slingshot_snapshot_count", slingshot_snapshot_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise ValueError("epochs must be a nonnegative integer")
    if not math.isfinite(threshold) or threshold <= 1:
        raise ValueError("threshold must be finite and greater than 1")
    if len(in_training) == 0 or len(in_training) != len(out_training):
        raise ValueError("training inputs and targets must have equal, nonzero lengths")
    if len(in_testing) == 0 or len(in_testing) != len(out_testing):
        raise ValueError("testing inputs and targets must have equal, nonzero lengths")

    directory = Path(prefix)
    directory.mkdir(parents=True, exist_ok=True)
    previous_loss = None
    dense_remaining = 0
    next_dense_epoch = None
    history = []

    for _ in range(epochs):
        trainer.test(in_testing, out_testing)
        with torch.no_grad():
            training_loss = loss_fn(out_training, trainer.m(in_training)).item()
        if not math.isfinite(training_loss) or training_loss < 0:
            raise ValueError("training loss must be finite and nonnegative")

        trainer.train(in_training, out_training)
        epoch = trainer.epoch
        # A rise from exactly zero is an infinite ratio; zero to zero is no spike.
        spike = previous_loss is not None and (
            (previous_loss == 0 and training_loss > 0)
            or (previous_loss > 0 and training_loss / previous_loss > threshold)
        )
        if spike:
            dense_remaining = slingshot_snapshot_count
            next_dense_epoch = epoch

        dense_due = dense_remaining > 0 and epoch == next_dense_epoch
        snapshot = None
        if epoch % snapshot_interval == 0 or dense_due:
            path = directory / f"{epoch}.pt"
            torch.save(trainer.m.state_dict(), path)
            snapshot = str(path)
        if dense_due:
            dense_remaining -= 1
            next_dense_epoch = epoch + slingshot_snapshot_interval

        history.append({"epoch": epoch, "training_loss": training_loss,
                        "spike": spike, "snapshot": snapshot})
        previous_loss = training_loss

    return history
