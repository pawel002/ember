import operator
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

import ember as em
from ember import Tensor


class TestTensorExhaustive:
    OPS = {
        "add": (operator.add, np.add),
        "sub": (operator.sub, np.subtract),
        "mul": (operator.mul, np.multiply),
        "div": (operator.truediv, np.true_divide),
        "gt": (operator.gt, np.greater),
        "lt": (operator.lt, np.less),
        "max": (em.max, np.maximum),
        "min": (em.min, np.minimum),
    }

    BROADCAST_OPS = ["add", "sub", "mul", "div"]
    UNARY_OPS = [
        ("neg", operator.neg, np.negative),
        ("exp", em.exp, np.exp),
        ("sin", em.sin, np.sin),
        ("cos", em.cos, np.cos),
        ("tan", em.tan, np.tan),
        ("ctg", em.ctg, lambda x: 1.0 / np.tan(x)),
        ("sinh", em.sinh, np.sinh),
        ("cosh", em.cosh, np.cosh),
        ("tanh", em.tanh, np.tanh),
        ("ctgh", em.ctgh, lambda x: 1.0 / np.tanh(x)),
        ("T", em.T, np.transpose),
    ]

    SHAPES = [(10,), (5, 5), (2, 5), (2, 3, 4)]
    BROADCAST_SHAPES = [
        ((32, 10), (10,)),  # Bias add
        ((10, 1), (1, 5)),  # Outer product
        ((8, 1, 64), (1, 10, 1)),  # 3D expansion
    ]
    SCALARS = [0.0, 1.0, -5.5, 100]

    def _gen_tensor(self, shape):
        np_t = np.random.randn(*shape).astype(np.float32)
        return Tensor.from_np(np_t), np_t

    def _assert_eq(self, t_res, np_res):
        assert t_res.shape == np_res.shape
        np.testing.assert_allclose(
            t_res.to_np(), np_res.astype(np.float32), rtol=1e-5, atol=1e-5
        )

    # --- Tests ---
    @pytest.mark.parametrize("op_name", OPS.keys())
    @pytest.mark.parametrize("shape", SHAPES)
    def test_binary_op_same_shape(self, op_name, shape):
        em_op, np_op = self.OPS[op_name]
        t_a, np_a = self._gen_tensor(shape)
        t_b, np_b = self._gen_tensor(shape)

        self._assert_eq(em_op(t_a, t_b), np_op(np_a, np_b))

    @pytest.mark.parametrize("op_name", BROADCAST_OPS)
    @pytest.mark.parametrize("shape_a, shape_b", BROADCAST_SHAPES)
    @pytest.mark.parametrize("reverse", [False, True])
    def test_binary_op_broadcast(self, op_name, shape_a, shape_b, reverse):
        em_op, np_op = self.OPS[op_name]
        t_a, np_a = self._gen_tensor(shape_a)
        t_b, np_b = self._gen_tensor(shape_b)

        if reverse:
            # test B op A
            self._assert_eq(em_op(t_b, t_a), np_op(np_b, np_a))
        else:
            # test A op B
            self._assert_eq(em_op(t_a, t_b), np_op(np_a, np_b))

    @pytest.mark.parametrize("op_name", OPS.keys())
    @pytest.mark.parametrize("scalar", SCALARS)
    @pytest.mark.parametrize("reverse", [False, True])
    def test_binary_op_scalar(self, op_name, scalar, reverse):
        if op_name == "div" and scalar == 0.0:
            pytest.skip("Skipping division by zero")

        em_op, np_op = self.OPS[op_name]
        t_a, np_a = self._gen_tensor((3, 3))

        if reverse:
            # scalar op tensor
            self._assert_eq(em_op(scalar, t_a), np_op(scalar, np_a))
        else:
            # tensor op scalar
            self._assert_eq(em_op(t_a, scalar), np_op(np_a, scalar))

    @pytest.mark.parametrize("name, em_op, np_op", UNARY_OPS)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_unary_op(self, name, em_op, np_op, shape):
        if name == "T" and len(shape) != 2:
            pytest.skip("Transpose requires 2D input")

        t_a, np_a = self._gen_tensor(shape)
        self._assert_eq(em_op(t_a), np_op(np_a))

    @pytest.mark.parametrize(
        "shape_a, shape_b, ctx",
        [
            ((4, 5), (5, 3), does_not_raise()),  # Valid
            ((1, 1), (1, 1), does_not_raise()),  # Valid Small
            ((4, 5), (4, 3), pytest.raises(ValueError)),  # Shape Mismatch
            ((4, 5, 2), (2, 3), pytest.raises(ValueError)),  # Rank Mismatch
            ((4, 5), None, pytest.raises(TypeError)),  # Invalid Type (simulated)
        ],
    )
    def test_matmul(self, shape_a, shape_b, ctx):
        t_a, np_a = self._gen_tensor(shape_a)

        if shape_b is None:
            with ctx:
                t_a @ 5
            return

        t_b, np_b = self._gen_tensor(shape_b)
        with ctx:
            self._assert_eq(t_a @ t_b, np_a @ np_b)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_sum(self, shape):
        t_a, np_a = self._gen_tensor(shape)

        assert abs(np.sum(np_a) - em.sum(t_a)) < 1e-5

        for axis in range(len(shape)):
            self._assert_eq(em.sum(t_a, axis=axis), np.sum(np_a, axis=axis))

    @pytest.mark.parametrize(
        "start, target, valid",
        [
            ((2, 3), (3, 2), True),
            ((2, 3), (6,), True),
            ((2, 3), (-1,), True),
            ((2, 3), (2, -1), True),
            ((2, 3), (4, 4), False),
            ((2, 3), (-1, -1), False),
        ],
    )
    def test_reshape(self, start, target, valid):
        np_a = np.arange(np.prod(start)).reshape(start).astype(np.float32)
        t_a = Tensor.from_np(np_a)

        if not valid:
            with pytest.raises(ValueError):
                t_a.reshape(target)
            return

        t_res = t_a.reshape(target)
        expected_np = np_a.reshape(target)
        self._assert_eq(t_res, expected_np)

        assert t_res is t_a
