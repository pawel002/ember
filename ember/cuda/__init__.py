"""CUDA-graph capture for fixed-shape training steps.

Capturing a training step into a CUDA graph and replaying it removes almost all
per-op launch and Python overhead -- the dominant cost for small networks.

    graph = ember.cuda.capture(step_fn)   # records step_fn once
    for _ in range(n):
        graph.replay()                    # near-zero overhead

Requirements: the step must have fixed shapes, allocate only through ember's
caching pool (warm up first so no cudaMalloc happens during capture), and read
no scalar back to the host mid-step (use Loss.gradient(), not the loss value).
Parameters are updated in place, so their buffers stay at stable addresses.

On the CPU backend these are no-ops (capture returns an inert graph).
"""

from __future__ import annotations

from collections.abc import Callable

from ember._core import (
    _begin_capture,
    _empty_cache,
    _end_capture,
    _get_matmul_tf32,
    _graph_destroy,
    _graph_launch,
    _set_matmul_tf32,
    _sync,
)


def set_matmul_tf32(enabled: bool) -> None:
    """Run cuBLAS GEMMs on the TF32 tensor cores instead of the fp32 pipeline.

    TF32 rounds the *inputs* of every matmul to a 10-bit mantissa and
    accumulates in fp32, for roughly 3x the GEMM throughput on Ampere and
    later. That is the standard trade for neural-network training (the same
    knob as ``torch.backends.cuda.matmul.allow_tf32``) but it is a real
    precision change, so it is off by default.
    """
    _set_matmul_tf32(bool(enabled))


def get_matmul_tf32() -> bool:
    """Whether TF32 matmuls are enabled (see :func:`set_matmul_tf32`)."""
    return _get_matmul_tf32()


def sync() -> None:
    """Block until all queued GPU work has finished."""
    _sync()


def empty_cache() -> None:
    """Release ember's cached device-memory pool back to the driver."""
    _empty_cache()


class Graph:
    """A captured, replayable training step."""

    def __init__(self, handle: int):
        self._handle = handle

    def replay(self) -> None:
        _graph_launch(self._handle)

    def __call__(self) -> None:
        _graph_launch(self._handle)

    def __del__(self):
        handle = getattr(self, "_handle", 0)
        if handle:
            _graph_destroy(handle)
            self._handle = 0


def capture(step: Callable[[], object], warmup: int = 3) -> Graph:
    """Run ``step`` a few times to warm up, then capture one invocation into a
    replayable :class:`Graph`."""
    for _ in range(warmup):
        step()
    sync()

    _begin_capture()
    step()
    return Graph(_end_capture())


__all__ = [
    "capture",
    "Graph",
    "sync",
    "empty_cache",
    "set_matmul_tf32",
    "get_matmul_tf32",
]
