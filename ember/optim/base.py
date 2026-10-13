from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ember.tensor import Tensor


class Optimizer(ABC):
    parameters: list[Tensor]

    @abstractmethod
    def __init__(self, parameters: list[Tensor], *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def apply(self, gradients: list[Tensor]) -> None:
        raise NotImplementedError

    # Per-parameter state buffers (e.g. Adam's means / variances), one list
    # entry per parameter, in parameter order. Subclasses with state override.
    def _state_tensors(self) -> dict[str, list[Tensor]]:
        return {}

    # Scalar state (e.g. the step counter) as plain Python numbers.
    def _state_scalars(self) -> dict[str, int | float]:
        return {}

    def _load_state_scalars(  # noqa: B027 -- optional hook, no-op by default
        self, scalars: dict[str, int | float]
    ) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        """Snapshot of the optimizer state as host (NumPy) arrays and scalars.

        Hyperparameters (lr, betas, ...) are not included: they come from the
        constructor, so a resumed run uses whatever the script passes in.
        """
        return {
            "type": type(self).__name__,
            "scalars": self._state_scalars(),
            "buffers": {
                name: [t.to_np() for t in tensors]
                for name, tensors in self._state_tensors().items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`.

        Buffers are copied into the existing device tensors in place, so cached
        ``_core`` lists and any captured CUDA graph stay valid. Everything is
        validated before anything is written.
        """
        if state["type"] != type(self).__name__:
            raise ValueError(
                f"optimizer state is for {state['type']}, "
                f"cannot load into {type(self).__name__}"
            )

        targets = self._state_tensors()
        buffers = state["buffers"]
        if set(buffers) != set(targets):
            raise ValueError(
                f"optimizer state buffers {sorted(buffers)} do not match "
                f"expected {sorted(targets)}"
            )
        for name, tensors in targets.items():
            arrays = buffers[name]
            if len(arrays) != len(tensors):
                raise ValueError(
                    f"optimizer state '{name}' has {len(arrays)} entries, "
                    f"expected {len(tensors)}"
                )
            for i, (t, a) in enumerate(zip(tensors, arrays, strict=True)):
                if tuple(np.shape(a)) != t.shape:
                    raise ValueError(
                        f"optimizer state '{name}[{i}]' has shape "
                        f"{tuple(np.shape(a))}, expected {t.shape}"
                    )

        for name, tensors in targets.items():
            for t, a in zip(tensors, buffers[name], strict=True):
                t.copy_from_numpy(a)
        self._load_state_scalars(state["scalars"])
