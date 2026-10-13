import numpy as np
import pytest

from ember import Tensor


class TestInplaceOps:
    SHAPES = [(10,), (5, 5), (2, 3, 4)]

    @pytest.mark.parametrize("shape", SHAPES)
    def test_inplace_tensor_same_shape(self, shape):
        a_np = np.random.randn(*shape).astype(np.float32)
        b_np = np.random.randn(*shape).astype(np.float32)

        t = Tensor.from_np(a_np)
        original = t  # identity must be preserved
        t += Tensor.from_np(b_np)
        assert t is original
        np.testing.assert_allclose(t.to_np(), a_np + b_np, rtol=1e-5, atol=1e-6)

        t -= Tensor.from_np(b_np)
        np.testing.assert_allclose(t.to_np(), a_np, rtol=1e-5, atol=1e-5)

        t *= Tensor.from_np(b_np)
        np.testing.assert_allclose(t.to_np(), a_np * b_np, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_inplace_scalar(self, shape):
        a_np = np.random.randn(*shape).astype(np.float32)
        t = Tensor.from_np(a_np)

        t += 2.0
        np.testing.assert_allclose(t.to_np(), a_np + 2.0, rtol=1e-5, atol=1e-6)
        t *= 3.0
        np.testing.assert_allclose(t.to_np(), (a_np + 2.0) * 3.0, rtol=1e-5, atol=1e-5)
        t /= 2.0
        np.testing.assert_allclose(
            t.to_np(), (a_np + 2.0) * 3.0 / 2.0, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("op", ["+=", "-=", "*=", "/="])
    def test_inplace_broadcast_keeps_the_buffer(self, op):
        # A broadcast operand is combined out of place, but the result is
        # copied back, so the tensor keeps its buffer (optimizers and captured
        # graphs hold on to it).
        a_np = np.random.randn(4, 5).astype(np.float32)
        b_np = np.random.rand(5).astype(np.float32) + 0.5

        t = Tensor.from_np(a_np)
        original, core = t, t._core
        alias = Tensor._from_core(core, t.shape, t.dtype)
        b = Tensor.from_np(b_np)
        if op == "+=":
            t += b
            expected = a_np + b_np
        elif op == "-=":
            t -= b
            expected = a_np - b_np
        elif op == "*=":
            t *= b
            expected = a_np * b_np
        else:
            t /= b
            expected = a_np / b_np
        assert t is original and t._core is core
        assert t.shape == (4, 5)
        np.testing.assert_allclose(alias.to_np(), expected, rtol=1e-5, atol=1e-5)

    def test_inplace_broadcast_cannot_grow_the_tensor(self):
        t = Tensor.from_np(np.ones(5, dtype=np.float32))
        with pytest.raises(ValueError, match="change the shape"):
            t += Tensor.from_np(np.ones((4, 5), dtype=np.float32))
        assert t.shape == (5,)

    def test_inplace_buffer_is_shared(self):
        # A true in-place op writes through to any alias holding the same _core.
        a_np = np.ones((3, 3), dtype=np.float32)
        t = Tensor.from_np(a_np)
        alias = Tensor._from_core(t._core, t.shape, t.dtype)

        t += 1.0  # scalar in-place mutates the shared buffer
        np.testing.assert_allclose(alias.to_np(), a_np + 1.0, rtol=1e-5, atol=1e-6)
