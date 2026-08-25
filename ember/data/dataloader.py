"""Batched iteration over a :class:`~ember.data.Dataset`."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from .collate import default_collate
from .dataset import Dataset
from .sampler import BatchSampler, RandomSampler, Sampler, SequentialSampler


class DataLoader:
    """Single-process batched loader over a map-style ``Dataset``.

    Each epoch iterates the batch sampler, fetches samples by index, and
    assembles them with ``collate_fn`` (default: :func:`default_collate`,
    which stacks NumPy and converts floating-point data to ``Tensor``).

    Args:
        dataset: the dataset to draw samples from.
        batch_size: samples per batch (mutually exclusive with
            ``batch_sampler``).
        shuffle: reshuffle indices every epoch (mutually exclusive with
            ``sampler``).
        sampler: custom index sampler; overrides ``shuffle``.
        batch_sampler: custom sampler yielding whole index batches; mutually
            exclusive with ``batch_size``, ``shuffle``, ``sampler`` and
            ``drop_last``.
        drop_last: drop the final incomplete batch so every batch has exactly
            ``batch_size`` samples. Required when batches feed fixed-shape
            buffers, e.g. CUDA-graph capture: allocate a fixed ``Tensor`` once
            and refill it each step with ``x.copy_from_numpy(batch_np)``.
        collate_fn: custom batch-assembly function.
        seed: seed for the shuffle generator (only valid with
            ``shuffle=True`` and no custom sampler).

    Loading is single-process by design; the per-batch work (index
    permutation, ``np.stack``, host->device copy) already runs in C/NumPy, so
    worker processes would add overhead rather than speed for the dataset
    sizes Ember targets.
    """

    dataset: Dataset
    batch_size: int
    drop_last: bool
    sampler: Sampler | None
    batch_sampler: BatchSampler | Sampler
    collate_fn: Callable[[list[Any]], Any]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Sampler | None = None,
        batch_sampler: Sampler | None = None,
        drop_last: bool = False,
        collate_fn: Callable[[list[Any]], Any] | None = None,
        seed: int | None = None,
    ):
        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")

        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive with "
                    "batch_size, shuffle, sampler, and drop_last"
                )

        if seed is not None and (sampler is not None or not shuffle):
            raise ValueError(
                "seed is only used together with shuffle=True and no custom sampler"
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn if collate_fn is not None else default_collate
        self.sampler = sampler

        if batch_sampler is None:
            if sampler is None:
                sampler = (
                    RandomSampler(dataset, seed=seed)
                    if shuffle
                    else SequentialSampler(dataset)
                )
                self.sampler = sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.batch_sampler = batch_sampler

    def __iter__(self) -> Iterator[Any]:
        for indices in self.batch_sampler:
            yield self.collate_fn([self.dataset[i] for i in indices])

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return len(self.batch_sampler)  # type: ignore[arg-type]
