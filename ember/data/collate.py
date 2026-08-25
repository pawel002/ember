"""Default batch-assembly (collate) logic for :class:`~ember.data.DataLoader`.

Batch assembly happens in NumPy (``np.stack``) because ember ``Tensor`` has
no stacking op; conversion to ``Tensor`` is a single backend copy afterwards.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ember.tensor import Tensor


def _collate_array(stacked: NDArray) -> Tensor | NDArray:
    """Float stacks become ``Tensor``; integer/bool stacks stay NumPy.

    The backend is float32-only and layers that consume integer data (e.g.
    ``nn.Embedding``) take NumPy index arrays directly, so integer batches are
    passed through as contiguous NumPy arrays instead of being lossily cast.
    """
    if stacked.dtype.kind == "f":
        return Tensor.from_np(stacked)
    return np.ascontiguousarray(stacked)


def default_collate(batch: list[Any]) -> Any:
    """Collate a list of samples into a batch.

    - NumPy arrays and ``Tensor``s are stacked along a new leading batch axis.
      Floating-point stacks are converted to ``Tensor``; integer/bool stacks
      stay NumPy (see :func:`_collate_array`).
    - Python ``float`` scalars become a 1-D ``Tensor``; ``int``/``bool``
      scalars become NumPy arrays.
    - tuples/lists are collated field-wise, dicts key-wise; strings pass
      through as a list.
    """
    elem = batch[0]

    if isinstance(elem, Tensor):
        return Tensor.from_np(np.stack([b.to_np() for b in batch]))
    if isinstance(elem, np.ndarray):
        return _collate_array(np.stack(batch))
    if isinstance(elem, bool | np.bool_):
        return np.asarray(batch)
    if isinstance(elem, int | np.integer):
        return np.asarray(batch)
    if isinstance(elem, float | np.floating):
        return Tensor.from_np(np.asarray(batch, dtype=np.float32))
    if isinstance(elem, str):
        return list(batch)
    if isinstance(elem, tuple):
        return tuple(
            default_collate(list(samples)) for samples in zip(*batch, strict=True)
        )
    if isinstance(elem, list):
        return [default_collate(list(samples)) for samples in zip(*batch, strict=True)]
    if isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}

    raise TypeError(
        f"default_collate does not support elements of type '{type(elem).__name__}'"
    )
