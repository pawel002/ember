"""MLP training-throughput benchmark: ember vs PyTorch (eager).

Measures steps/second for a full training step (forward + loss + backward +
optimizer) on small, fixed-shape MLPs -- the regime ember targets. Run with:

    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    uv run --group bench python benchmarks/bench_mlp.py

PyTorch is a benchmark-only dependency (the `bench` group); ember never imports
it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    name: str
    batch: int
    in_dim: int
    hidden: int
    out_dim: int
    depth: int  # number of Linear layers
    warmup: int = 50
    iters: int = 300


CONFIGS = [
    Config("tiny", batch=64, in_dim=64, hidden=64, out_dim=1, depth=2),
    Config("small", batch=256, in_dim=128, hidden=256, out_dim=10, depth=3),
    Config("medium", batch=512, in_dim=256, hidden=512, out_dim=10, depth=3),
]


def make_data(cfg: Config, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((cfg.batch, cfg.in_dim)).astype(np.float32)
    y = rng.standard_normal((cfg.batch, cfg.out_dim)).astype(np.float32)
    return x, y


def _hidden_sizes(cfg: Config) -> list[tuple[int, int]]:
    """(in, out) for each Linear layer."""
    sizes = []
    prev = cfg.in_dim
    for i in range(cfg.depth):
        out = cfg.out_dim if i == cfg.depth - 1 else cfg.hidden
        sizes.append((prev, out))
        prev = out
    return sizes


# --------------------------------------------------------------------------- #
# ember
# --------------------------------------------------------------------------- #
def bench_ember(cfg: Config, x_np, y_np, opt_name="adam", use_graph=False) -> float:
    import ember as em
    import ember.loss as loss
    import ember.nn as nn
    import ember.optim as optim

    em.random.seed(0)
    layers = []
    for i, (a, b) in enumerate(_hidden_sizes(cfg)):
        layers.append(nn.Linear(a, b))
        if i < cfg.depth - 1:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    if opt_name == "sgd":
        opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    else:
        opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = loss.MSELoss()

    x = em.Tensor(x_np)
    target = em.Tensor(y_np)

    def step():
        pred = model(x, training=True)
        # gradient() computes dL/dpred without materializing the scalar loss,
        # so there is no device-to-host sync in the hot loop (matching torch,
        # which does not call loss.item() here either).
        model.backward(criterion.gradient(pred, target))
        opt.apply(model.gradients())

    if use_graph:
        graph = em.cuda.capture(step, warmup=cfg.warmup)
        em.cuda.sync()
        t0 = time.perf_counter()
        for _ in range(cfg.iters):
            graph.replay()
        em.cuda.sync()
        t1 = time.perf_counter()
        return cfg.iters / (t1 - t0)

    for _ in range(cfg.warmup):
        step()
    em.cuda.sync()

    t0 = time.perf_counter()
    for _ in range(cfg.iters):
        step()
    em.cuda.sync()
    t1 = time.perf_counter()

    return cfg.iters / (t1 - t0)


# --------------------------------------------------------------------------- #
# torch (eager)
# --------------------------------------------------------------------------- #
def bench_torch(cfg: Config, x_np, y_np, opt_name="adam") -> float:
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F

    assert torch.cuda.is_available(), "benchmark requires a CUDA-capable torch"
    dev = torch.device("cuda")
    torch.manual_seed(0)

    layers = []
    for i, (a, b) in enumerate(_hidden_sizes(cfg)):
        layers.append(tnn.Linear(a, b))
        if i < cfg.depth - 1:
            layers.append(tnn.ReLU())
    model = tnn.Sequential(*layers).to(dev)
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.from_numpy(x_np).to(dev)
    target = torch.from_numpy(y_np).to(dev)

    def step():
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, target)
        loss.backward()
        opt.step()

    for _ in range(cfg.warmup):
        step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(cfg.iters):
        step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return cfg.iters / (t1 - t0)


def main() -> int:
    # Table 1: apples-to-apples eager training, same optimizer (Adam).
    print("== eager, Adam (ember vs torch) ==")
    print(
        f"{'config':>8} | {'ember (it/s)':>13} | {'torch (it/s)':>13} | {'speedup':>8}"
    )
    print("-" * 54)

    worst = float("inf")
    for cfg in CONFIGS:
        x_np, y_np = make_data(cfg)
        em_ips = bench_ember(cfg, x_np, y_np)
        th_ips = bench_torch(cfg, x_np, y_np)
        speedup = em_ips / th_ips
        worst = min(worst, speedup)
        flag = "OK" if speedup >= 1.0 else "SLOWER"
        print(
            f"{cfg.name:>8} | {em_ips:13.0f} | {th_ips:13.0f} | {speedup:7.2f}x {flag}"
        )
    print("-" * 54)
    print(f"worst-case ember/torch speedup (eager Adam): {worst:.2f}x\n")

    # Table 2: CUDA-graph capture (SGD, which is exact under capture).
    print("== +CUDA graph, SGD (ember eager / ember graph / torch eager) ==")
    print(
        f"{'config':>8} | {'em-eager':>10} | {'em-graph':>10} | "
        f"{'torch':>10} | {'graph/torch':>11}"
    )
    print("-" * 62)
    for cfg in CONFIGS:
        x_np, y_np = make_data(cfg)
        em_eager = bench_ember(cfg, x_np, y_np, opt_name="sgd", use_graph=False)
        em_graph = bench_ember(cfg, x_np, y_np, opt_name="sgd", use_graph=True)
        th = bench_torch(cfg, x_np, y_np, opt_name="sgd")
        print(
            f"{cfg.name:>8} | {em_eager:10.0f} | {em_graph:10.0f} | "
            f"{th:10.0f} | {em_graph / th:10.2f}x"
        )

    return 0 if worst >= 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
