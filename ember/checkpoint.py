"""Saving and loading model / optimizer checkpoints.

    ember.save("ckpt.npz", model, opt, step=step)
    meta = ember.load("ckpt.npz", model, opt)   # -> {"step": step}

``model`` is anything with a ``parameters()`` method (any ``nn.Layer``, or a
hand-written model class like tiny-gpt's ``GPT``). Parameters are matched by
position in that list, so the model must be constructed with the same
architecture before loading.

Loading copies into the existing device buffers in place: optimizer references
and captured CUDA graphs keep pointing at the right memory, and nothing has to
be rebuilt after a restore.

The file is a plain ``.npz`` archive (float32 arrays plus a JSON header) read
with ``allow_pickle=False``, so loading an untrusted checkpoint cannot execute
code. It is written to a temporary file and renamed into place, so an
interrupted save never leaves a truncated checkpoint behind.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Protocol, cast

import numpy as np

from ember.optim.base import Optimizer
from ember.tensor import Tensor

FORMAT_VERSION = 1
_META_KEY = "__meta__"


class HasParameters(Protocol):
    def parameters(self) -> list[Tensor]: ...


def save(
    path: str | os.PathLike,
    model: HasParameters | None = None,
    optimizer: Optimizer | None = None,
    **metadata: Any,
) -> None:
    """Write a checkpoint of ``model`` and/or ``optimizer`` to ``path``.

    Extra keyword arguments (step, epoch, config, ...) are stored as JSON and
    returned by :func:`load`, so they must be JSON-serializable.
    """
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"format_version": FORMAT_VERSION, "metadata": metadata}

    if model is not None:
        params = model.parameters()
        meta["num_parameters"] = len(params)
        for i, p in enumerate(params):
            arrays[f"model/{i}"] = p.to_np()

    if optimizer is not None:
        state = optimizer.state_dict()
        meta["optimizer"] = {
            "type": state["type"],
            "scalars": state["scalars"],
            "buffers": {name: len(a) for name, a in state["buffers"].items()},
        }
        for name, buf in state["buffers"].items():
            for i, a in enumerate(buf):
                arrays[f"optim/{name}/{i}"] = a

    try:
        header = json.dumps(meta)
    except TypeError as e:
        raise TypeError(f"checkpoint metadata must be JSON-serializable: {e}") from e
    arrays[_META_KEY] = np.frombuffer(header.encode(), dtype=np.uint8)

    path = os.fspath(path)
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            # A file object, so no ".npz" suffix is appended. (The cast keeps
            # mypy from matching the arrays against savez's allow_pickle flag.)
            np.savez(f, **cast(dict[str, Any], arrays))
        # mkstemp creates the file 0600; give it normal umask-based permissions
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def load(
    path: str | os.PathLike,
    model: HasParameters | None = None,
    optimizer: Optimizer | None = None,
) -> dict[str, Any]:
    """Restore ``model`` and/or ``optimizer`` from a checkpoint written by
    :func:`save`, and return the extra metadata that was saved with it.

    Every shape is checked before anything is written, so a mismatched
    checkpoint raises ``ValueError`` and leaves the model untouched.
    """
    with np.load(os.fspath(path), allow_pickle=False) as f:
        meta = json.loads(f[_META_KEY].tobytes().decode())
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported checkpoint format version {meta.get('format_version')}"
            )

        params: list[Tensor] = []
        model_arrays: list[np.ndarray] = []
        if model is not None:
            if "num_parameters" not in meta:
                raise ValueError("checkpoint does not contain model parameters")
            params = model.parameters()
            if len(params) != meta["num_parameters"]:
                raise ValueError(
                    f"checkpoint has {meta['num_parameters']} parameters, "
                    f"model has {len(params)}"
                )
            model_arrays = [f[f"model/{i}"] for i in range(len(params))]
            for i, (p, a) in enumerate(zip(params, model_arrays, strict=True)):
                if tuple(a.shape) != p.shape:
                    raise ValueError(
                        f"parameter {i} has shape {tuple(a.shape)} in the "
                        f"checkpoint, but {p.shape} in the model"
                    )

        optim_state: dict[str, Any] | None = None
        if optimizer is not None:
            if "optimizer" not in meta:
                raise ValueError("checkpoint does not contain optimizer state")
            om = meta["optimizer"]
            optim_state = {
                "type": om["type"],
                "scalars": om["scalars"],
                "buffers": {
                    name: [f[f"optim/{name}/{i}"] for i in range(n)]
                    for name, n in om["buffers"].items()
                },
            }

    # Optimizer first: it validates fully before writing, and the model
    # shapes were checked above, so a failure here leaves both untouched.
    if optimizer is not None and optim_state is not None:
        optimizer.load_state_dict(optim_state)
    for p, a in zip(params, model_arrays, strict=True):
        p.copy_from_numpy(a)

    return meta["metadata"]
