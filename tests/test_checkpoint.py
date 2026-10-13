import numpy as np
import pytest

import ember as em
import ember.nn as nn
import ember.optim as optim
from ember import Tensor


def _mlp():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.LayerNorm(8),
        nn.Linear(8, 3),
    )


def _params_np(model):
    return [p.to_np() for p in model.parameters()]


def _train(model, opt, steps, seed):
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        x = Tensor.from_np(rng.standard_normal((5, 4)).astype(np.float32))
        y = model(x)
        model.backward(y * 0.1)
        opt.apply(model.gradients())


def test_model_roundtrip_in_place(tmp_path):
    src, dst = _mlp(), _mlp()
    cores = [p._core for p in dst.parameters()]

    em.save(tmp_path / "m.npz", src)
    em.load(tmp_path / "m.npz", dst)

    for a, b in zip(_params_np(src), _params_np(dst), strict=True):
        np.testing.assert_array_equal(a, b)
    # buffers were updated in place, not replaced
    assert all(p._core is c for p, c in zip(dst.parameters(), cores, strict=True))


def test_transformer_roundtrip(tmp_path):
    src = nn.TransformerEncoder(2, 32, 4, causal=True)
    dst = nn.TransformerEncoder(2, 32, 4, causal=True)
    em.save(tmp_path / "t.npz", src)
    em.load(tmp_path / "t.npz", dst)
    for a, b in zip(_params_np(src), _params_np(dst), strict=True):
        np.testing.assert_array_equal(a, b)


def test_duck_typed_model(tmp_path):
    class Model:
        def __init__(self):
            self.a, self.b = nn.Linear(3, 3), nn.Embedding(10, 3)

        def parameters(self):
            return self.a.parameters() + self.b.parameters()

    src, dst = Model(), Model()
    em.save(tmp_path / "d.npz", src)
    em.load(tmp_path / "d.npz", dst)
    for a, b in zip(_params_np(src), _params_np(dst), strict=True):
        np.testing.assert_array_equal(a, b)


def test_metadata_roundtrip(tmp_path):
    em.save(tmp_path / "m.npz", _mlp(), step=42, config={"lr": 1e-3, "tag": "x"})
    assert em.load(tmp_path / "m.npz") == {
        "step": 42,
        "config": {"lr": 1e-3, "tag": "x"},
    }


def test_non_json_metadata_raises(tmp_path):
    with pytest.raises(TypeError, match="JSON"):
        em.save(tmp_path / "m.npz", _mlp(), bad=object())
    assert not list(tmp_path.iterdir())  # no partial / temp file left behind


def test_exact_path_no_suffix_added(tmp_path):
    em.save(tmp_path / "model.ckpt", _mlp())
    assert [p.name for p in tmp_path.iterdir()] == ["model.ckpt"]


@pytest.mark.parametrize(
    "make_opt",
    [
        lambda p: optim.SGD(p, lr=0.05, momentum=0.9),
        lambda p: optim.Adam(p, lr=0.01),
        lambda p: optim.Adam(p, lr=0.01, foreach=True),
        lambda p: optim.Adam(p, lr=0.01, capturable=True),
        lambda p: optim.AdamW(p, lr=0.01, weight_decay=0.1),
        lambda p: optim.AdamW(p, lr=0.01, capturable=True),
    ],
    ids=["sgd", "adam", "adam-foreach", "adam-capturable", "adamw", "adamw-capturable"],
)
def test_resume_matches_uninterrupted(tmp_path, make_opt):
    """Train 3 steps, checkpoint, train 3 more == load into a fresh model +
    optimizer and train the same 3 more."""
    em.random.seed(0)
    model = _mlp()
    opt = make_opt(model.parameters())
    _train(model, opt, 3, seed=1)
    em.save(tmp_path / "c.npz", model, opt, step=3)
    _train(model, opt, 3, seed=2)
    expected = _params_np(model)

    em.random.seed(123)  # different init, overwritten by the load
    resumed = _mlp()
    ropt = make_opt(resumed.parameters())
    assert em.load(tmp_path / "c.npz", resumed, ropt) == {"step": 3}
    _train(resumed, ropt, 3, seed=2)

    for a, b in zip(expected, _params_np(resumed), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


def test_adam_step_counter_crosses_capturable_modes(tmp_path):
    model = _mlp()
    opt = optim.Adam(model.parameters(), capturable=True)
    _train(model, opt, 4, seed=0)
    em.save(tmp_path / "c.npz", model, opt)

    eager = optim.Adam(_mlp().parameters())
    em.load(tmp_path / "c.npz", optimizer=eager)
    assert eager.t == 4


def test_shape_mismatch_leaves_model_untouched(tmp_path):
    em.save(tmp_path / "m.npz", _mlp())
    other = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.LayerNorm(8), nn.Linear(8, 5))
    before = _params_np(other)
    with pytest.raises(ValueError, match="shape"):
        em.load(tmp_path / "m.npz", other)
    for a, b in zip(before, _params_np(other), strict=True):
        np.testing.assert_array_equal(a, b)


def test_parameter_count_mismatch(tmp_path):
    em.save(tmp_path / "m.npz", _mlp())
    with pytest.raises(ValueError, match="parameters"):
        em.load(tmp_path / "m.npz", nn.Linear(4, 8))


def test_optimizer_type_mismatch_leaves_both_untouched(tmp_path):
    model = _mlp()
    em.save(tmp_path / "c.npz", model, optim.SGD(model.parameters()))
    fresh = _mlp()
    before = _params_np(fresh)
    with pytest.raises(ValueError, match="SGD"):
        em.load(tmp_path / "c.npz", fresh, optim.Adam(fresh.parameters()))
    for a, b in zip(before, _params_np(fresh), strict=True):
        np.testing.assert_array_equal(a, b)


def test_missing_sections(tmp_path):
    model = _mlp()
    em.save(tmp_path / "m.npz", model)
    with pytest.raises(ValueError, match="optimizer"):
        em.load(tmp_path / "m.npz", optimizer=optim.SGD(model.parameters()))

    em.save(tmp_path / "o.npz", optimizer=optim.SGD(model.parameters()))
    with pytest.raises(ValueError, match="model"):
        em.load(tmp_path / "o.npz", model)
