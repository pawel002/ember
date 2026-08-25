"""Sampling strategies for :class:`~ember.data.DataLoader`.

A sampler decides the order in which dataset indices are visited. Randomness
uses a ``numpy.random.Generator``; passing ``seed`` makes the sequence of
epoch orders reproducible (each epoch still gets a fresh permutation, like
PyTorch).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sized

import numpy as np


class Sampler(ABC):
    """Base class for samplers. Iterating yields dataset indices."""

    @abstractmethod
    def __iter__(self) -> Iterator:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Yields indices ``0 .. len(data_source) - 1`` in order."""

    def __init__(self, data_source: Sized):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """Yields indices in random order.

    Without ``replacement`` each epoch is a fresh permutation of all indices.
    With ``replacement=True``, ``num_samples`` indices are drawn uniformly
    with replacement per epoch (``num_samples`` defaults to the dataset size).

    The internal generator is seeded once at construction; iterating the same
    sampler twice gives different (but reproducible across identical seeds)
    permutations.
    """

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: int | None = None,
        seed: int | None = None,
    ):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = len(data_source) if num_samples is None else num_samples
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.replacement:
            return iter(self._rng.integers(0, n, size=self.num_samples).tolist())
        return iter(self._rng.permutation(n).tolist())

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler):
    """Groups indices from another sampler into batches of ``batch_size``.

    The final partial batch is yielded unless ``drop_last=True`` — keep
    ``drop_last=True`` when batches feed fixed-shape buffers (e.g. CUDA-graph
    capture with ``Tensor.copy_from_numpy``).
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
