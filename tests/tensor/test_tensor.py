import pytest
import numpy as np
import operator
from contextlib import nullcontext as does_not_raise

import ember as em
from ember import Tensor


class TestTensorExhaustive:
    BINARY_OPS = [
        ("add", operator.add, np.add),
        ("sub", operator.sub, np.subtract),
        ("mul", operator.mul, np.multiply),
        ("gt", operator.gt, np.greater),
        ("max", em.max, np.maximum),
        ("min", em.min, np.minimum),
    ]
    REVERSABLE_OPS = [
        ("radd", operator.add, np.add),
        ("rsub", operator.sub, np.subtract),
        ("rmul", operator.mul, np.multiply),
    ]

    def _assert_tensor_eq_np(self, tensor_res, np_res):
        assert (
            tensor_res.shape == np_res.shape
        ), f"Shape Mismatch: Tensor {tensor_res.shape} vs NP {np_res.shape}"

        tensor_data = tensor_res.to_np()
        np.testing.assert_allclose(
            tensor_data, np_res.astype(np.float32), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("method_name, py_op, np_op", BINARY_OPS)
    @pytest.mark.parametrize("shape", [(10,), (5, 5), (2, 3, 4)])
    def test_binary_op_tensor_vs_tensor(self, method_name, py_op, np_op, shape):
        np_a = np.random.randn(*shape).astype(np.float32)
        np_b = np.random.randn(*shape).astype(np.float32)

        t_a = Tensor.from_np(np_a)
        t_b = Tensor.from_np(np_b)

        t_res = py_op(t_a, t_b)
        np_res = np_op(np_a, np_b)

        self._assert_tensor_eq_np(t_res, np_res)

    @pytest.mark.parametrize("method_name, py_op, np_op", BINARY_OPS)
    @pytest.mark.parametrize("scalar", [0.0, 1.0, -5.5, 100])
    def test_binary_op_tensor_vs_scalar(self, method_name, py_op, np_op, scalar):
        shape = (3, 3)
        np_a = np.random.randn(*shape).astype(np.float32)
        t_a = Tensor.from_np(np_a)

        t_res = py_op(t_a, scalar)
        np_res = np_op(np_a, scalar)

        self._assert_tensor_eq_np(t_res, np_res)

    @pytest.mark.parametrize("method_name, py_op, np_op", REVERSABLE_OPS)
    @pytest.mark.parametrize("scalar", [0.0, 1.0, -5.5, 100])
    def test_reverse_binary_op_scalar_vs_tensor(
        self, method_name, py_op, np_op, scalar
    ):
        shape = (3, 3)
        np_a = np.random.randn(*shape).astype(np.float32)
        t_a = Tensor.from_np(np_a)

        t_res = py_op(scalar, t_a)
        np_res = np_op(scalar, np_a)

        self._assert_tensor_eq_np(t_res, np_res)

    @pytest.mark.parametrize(
        "shape_a, shape_b, expectation",
        [
            ((4, 5), (5, 3), does_not_raise()),
            ((1, 1), (1, 1), does_not_raise()),
            ((4, 5), (4, 3), pytest.raises(ValueError)),
            ((4, 5, 2), (2, 3), pytest.raises(ValueError)),
            ((4, 5), 5, pytest.raises(TypeError)),
        ],
    )
    def test_matmul(self, shape_a, shape_b, expectation):
        if isinstance(shape_b, int):
            np_a = np.random.randn(*shape_a).astype(np.float32)
            t_a = Tensor.from_np(np_a)
            with expectation:
                _ = t_a @ shape_b
            return

        np_a = np.random.randn(*shape_a).astype(np.float32)
        np_b = np.random.randn(*shape_b).astype(np.float32)
        t_a = Tensor.from_np(np_a)
        t_b = Tensor.from_np(np_b)

        with expectation:
            t_res = t_a @ t_b
            np_res = np_a @ np_b
            self._assert_tensor_eq_np(t_res, np_res)

    def test_negate(self):
        np_a = np.random.randn(5, 5).astype(np.float32)
        t_a = Tensor.from_np(np_a)

        t_res = -t_a
        np_res = -np_a

        self._assert_tensor_eq_np(t_res, np_res)

    @pytest.mark.parametrize(
        "start_shape, target_shape, valid",
        [
            ((2, 3), (3, 2), True),
            ((2, 3), (6,), True),
            ((2, 3), (-1,), True),
            ((2, 3), (2, -1), True),
            ((2, 3), (4, 4), False),
            ((2, 3), (-1, -1), False),
            ((6,), (2, 3), True),
        ],
    )
    def test_reshape(self, start_shape, target_shape, valid):
        np_a = np.arange(np.prod(start_shape)).reshape(start_shape).astype(np.float32)
        t_a = Tensor.from_np(np_a)

        if valid:
            t_res = t_a.reshape(target_shape)
            expected_np = np_a.reshape(target_shape)

            assert t_res.shape == expected_np.shape
            self._assert_tensor_eq_np(t_res, expected_np)
        else:
            with pytest.raises(ValueError):
                t_a.reshape(target_shape)

    def test_reshape_is_inplace_and_returns_self(self):
        t = Tensor.from_np(np.zeros((2, 3)))
        original_id = id(t)

        t_returned = t.reshape((6,))

        assert id(t_returned) == original_id, "Reshape should return self"
        assert t.shape == (6,), "Reshape should modify shape in-place"

    def test_shape_mismatch_raises_error(self):
        t_a = Tensor.from_np(np.zeros((2, 2)))
        t_b = Tensor.from_np(np.zeros((3, 3)))

        for method, py_op, _ in self.BINARY_OPS:
            with pytest.raises(ValueError, match="Shape mismatch"):
                if method.startswith("__"):
                    py_op(t_a, t_b)
                else:
                    py_op(t_a, t_b)
