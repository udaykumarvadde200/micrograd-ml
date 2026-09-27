import numpy as np


def create_batches(
    X,
    y,
    batch_size=None,
    shuffle=True,
    seed=42
):
    """
    Create batches from a dataset.

    batch_size=1
        Sample-by-sample training.

    batch_size=32
        Mini-batch training.

    batch_size=None
        Full-batch training.
    """

    n_samples = len(X)

    if batch_size is None:
        yield X, y
        return

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be a positive integer or None"
        )

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(
        0,
        n_samples,
        batch_size
    ):

        batch_indices = indices[
            start:start + batch_size
        ]

        yield (
            X[batch_indices],
            y[batch_indices]
        )