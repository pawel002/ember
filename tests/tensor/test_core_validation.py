"""The _core bindings check every caller-supplied shape before a kernel runs,
so a wrong argument raises instead of reading or writing out of bounds."""

import numpy as np
import pytest

import ember as em
from ember import Tensor, _core


def _t(*shape):
    return Tensor.from_np(np.ones(shape, dtype=np.float32))._core


@pytest.mark.parametrize(
    "call",
    [
        lambda: _core._transpose(_t(6), 2, 4),
        lambda: _core._matmul(_t(2, 3), _t(4, 5), 2, 5, 3),
        lambda: _core._matmul_bias(_t(2, 3), _t(3, 5), _t(4), 2, 5, 3),
        lambda: _core._gelu_bwd(_t(4), _t(5)),
        lambda: _core._sum_axis(_t(6), (2, 4), 0),
        lambda: _core._max_axis(_t(6), (2, 3), 2),
        lambda: _core._softmax_fwd(_t(6), (2, 3), -1),
        lambda: _core._softmax_rows(_t(6), 2, 4),
        lambda: _core._softmax_rows_causal(_t(6), 2, 3, 0),
        lambda: _core._layernorm_fwd(_t(6), _t(3), _t(2), 2, 3, 1e-5),
        lambda: _core._permute_0213(_t(6), 1, 2, 3, 2),
        lambda: _core._bmm(_t(6), _t(6), 1, 2, 2, 4, 0, 0, 1.0),
        lambda: _core._adam_step(
            _t(4), _t(4), _t(3), _t(4), 0.1, 0.9, 0.1, 0.99, 0.01, 1e-8, 1.0, 1.0, 0.0
        ),
        lambda: _core._sgd_step(_t(4), _t(5), _t(4), 0.1, 0.0),
        lambda: _core._adam_step_group(
            [_t(4)],
            [_t(4)],
            [_t(4)],
            [_t(2)],
            0.1,
            0.9,
            0.1,
            0.99,
            0.01,
            1e-8,
            1.0,
            1.0,
            0.0,
        ),
        lambda: _core._attention_fwd(
            _t(2, 4, 16), _t(2, 4, 16), _t(2, 4, 16), 2, 4, 4, 16, 1.0, False, 1, 32
        ),
        lambda: _core._embedding_bwd(_t(3, 4), np.zeros(2, np.int32), 5, 4),
    ],
)
def test_shape_mismatch_raises(call):
    with pytest.raises(ValueError):
        call()


def test_broadcast_rank_limit():
    a = _t(*([1] * 9))
    with pytest.raises(ValueError, match="at most 8"):
        _core._add_broadcasted(a, a, (1,) * 9, (0,) * 9, (0,) * 9)


def test_broadcast_strides_must_stay_inside_operands():
    with pytest.raises(ValueError, match="exceed"):
        _core._add_broadcasted(_t(2, 3), _t(3), (2, 3), (3, 1), (3, 1))


@pytest.mark.parametrize("bad", [[-1, 0], [0, 7]])
def test_embedding_index_out_of_range(bad):
    weight = _t(7, 4)
    with pytest.raises(IndexError, match="out of range"):
        _core._embedding_fwd(weight, np.array(bad, np.int32), 4)


def test_embedding_rejects_float_indices():
    with pytest.raises(TypeError, match="int32"):
        _core._embedding_fwd(_t(7, 4), np.zeros(3, np.float32), 4)


class TestEmptyTensors:
    def test_roundtrip(self):
        t = Tensor(np.zeros((0, 3), np.float32))
        assert t.shape == (0, 3)
        assert t.to_np().shape == (0, 3)
        assert t.to_list() == []

    def test_elementwise_and_sum(self):
        a = Tensor(np.zeros((0, 3), np.float32))
        assert (a + 1.0).to_np().shape == (0, 3)
        assert (a * a).to_np().shape == (0, 3)
        assert em.exp(a).to_np().shape == (0, 3)
        a += 2.0
        assert em.sum(a) == 0.0
        assert em.sum(a, axis=0).to_np().tolist() == [0.0, 0.0, 0.0]

    def test_matmul_with_empty_inner_dim_is_zeros(self):
        a = Tensor(np.zeros((2, 0), np.float32))
        b = Tensor(np.zeros((0, 3), np.float32))
        np.testing.assert_array_equal((a @ b).to_np(), np.zeros((2, 3)))

    def test_random_empty(self):
        assert em.random.zeros((0, 4)).to_np().shape == (0, 4)

    def test_max_over_empty_axis_raises(self):
        with pytest.raises(ValueError, match="empty axis"):
            em.amax(Tensor(np.zeros((2, 0), np.float32)), axis=1)
