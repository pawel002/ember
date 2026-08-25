"""Map-style dataset abstractions, modeled after ``torch.utils.data``.

Datasets yield individual samples as NumPy arrays / Python scalars; batching
and conversion to ``Tensor`` happen in :class:`~ember.data.DataLoader` via a
``collate_fn`` (see :mod:`ember.data.collate`).
"""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import accumulate
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ember.tensor import Tensor

ArrayLike: TypeAlias = NDArray[Any] | Tensor


class Dataset(ABC):
    """Abstract map-style dataset.

    Subclasses must implement ``__getitem__`` (fetch a sample by index) and
    ``__len__``. Adding two datasets with ``+`` returns a ``ConcatDataset``.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def __add__(self, other: Dataset) -> ConcatDataset:
        return ConcatDataset([self, other])


class TensorDataset(Dataset):
    """Dataset wrapping one or more equal-length arrays.

    Each argument is a NumPy array or ``Tensor`` whose first dimension is the
    sample axis; ``dataset[i]`` returns a tuple holding the i-th row of each
    array (as NumPy). All arrays must agree on the first dimension.
    """

    def __init__(self, *arrays: ArrayLike):
        if not arrays:
            raise ValueError("TensorDataset requires at least one array")

        self.arrays: tuple[NDArray, ...] = tuple(
            a.to_np() if isinstance(a, Tensor) else np.asarray(a) for a in arrays
        )

        n = len(self.arrays[0])
        if any(len(a) != n for a in self.arrays):
            raise ValueError(
                "All arrays must have the same size in the first dimension"
            )

    def __getitem__(self, index: int) -> tuple[NDArray, ...]:
        return tuple(a[index] for a in self.arrays)

    def __len__(self) -> int:
        return len(self.arrays[0])


class Subset(Dataset):
    """A view of ``dataset`` restricted to ``indices``."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)


class ConcatDataset(Dataset):
    """A dataset concatenating several datasets of any kind."""

    def __init__(self, datasets: Sequence[Dataset]):
        datasets = list(datasets)
        if not datasets:
            raise ValueError("ConcatDataset requires at least one dataset")
        if any(len(d) == 0 for d in datasets):
            raise ValueError("ConcatDataset does not support empty datasets")
        self.datasets = datasets
        self.cumulative_sizes: list[int] = list(accumulate(len(d) for d in datasets))

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(
                f"index {index} out of range for ConcatDataset of length {len(self)}"
            )
        ds_idx = bisect.bisect_right(self.cumulative_sizes, index)
        sample_idx = index if ds_idx == 0 else index - self.cumulative_sizes[ds_idx - 1]
        return self.datasets[ds_idx][sample_idx]


def random_split(
    dataset: Dataset, lengths: Sequence[int], seed: int | None = None
) -> list[Subset]:
    """Randomly split ``dataset`` into disjoint ``Subset``s of ``lengths``.

    ``lengths`` must sum to ``len(dataset)``. Pass ``seed`` for a
    reproducible split.
    """
    if any(n < 0 for n in lengths):
        raise ValueError(f"lengths must be non-negative, got {list(lengths)}")
    if sum(lengths) != len(dataset):
        raise ValueError(
            f"lengths must sum to len(dataset): {sum(lengths)} != {len(dataset)}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()

    subsets = []
    offset = 0
    for n in lengths:
        subsets.append(Subset(dataset, indices[offset : offset + n]))
        offset += n
    return subsets
