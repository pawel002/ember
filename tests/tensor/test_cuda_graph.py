import numpy as np
import pytest

import ember as em
import ember.loss as loss
import ember.nn as nn
import ember.optim as optim
from ember import Tensor


def _graphs_available() -> bool:
    # On the CPU backend capture is a no-op and returns a null (0) handle.
    probe = em.cuda.capture(lambda: None)
    return probe._handle != 0


pytestmark = pytest.mark.skipif(
    not _graphs_available(), reason="CUDA graphs unavailable (CPU backend)"
)


def test_graph_replay_matches_eager_inplace():
    n = 5
    tg = Tensor.from_np(np.zeros((32,), dtype=np.float32))
    graph = em.cuda.capture(lambda: tg.__iadd__(2.0), warmup=3)
    for _ in range(n):
        graph.replay()
    em.cuda.sync()

    # capture ran 3 warmup steps; replays add n more => 3 + n increments of 2.
    np.testing.assert_allclose(tg.to_np(), 2.0 * (3 + n), atol=1e-6)


def test_graph_training_matches_eager():
    warmup, n = 5, 25

    def build():
        em.random.seed(0)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        return model, opt, loss.MSELoss()

    x_np = np.random.randn(32, 8).astype(np.float32)
    y_np = np.random.randn(32, 4).astype(np.float32)

    # --- eager reference: warmup + n steps ---
    m_e, o_e, c_e = build()
    xe, ye = Tensor(x_np), Tensor(y_np)

    def step_e():
        m_e.backward(c_e.gradient(m_e(xe, training=True), ye))
        o_e.apply(m_e.gradients())

    for _ in range(warmup + n):
        step_e()
    em.cuda.sync()
    w_eager = m_e.parameters()[0].to_np().copy()

    # --- captured graph: warmup during capture() + n replays ---
    m_g, o_g, c_g = build()
    xg, yg = Tensor(x_np), Tensor(y_np)

    def step_g():
        m_g.backward(c_g.gradient(m_g(xg, training=True), yg))
        o_g.apply(m_g.gradients())

    graph = em.cuda.capture(step_g, warmup=warmup)
    for _ in range(n):
        graph.replay()
    em.cuda.sync()
    w_graph = m_g.parameters()[0].to_np()

    np.testing.assert_allclose(w_graph, w_eager, rtol=1e-4, atol=1e-5)
