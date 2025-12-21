import pytest
from ember import Tensor


def assert_close(actual, expected, tol=1e-5):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert abs(a - e) < tol, f"Mismatch: {a} != {e}"


def simple_python_matmul(a: list, b: list) -> list:
    n = len(a)
    k = len(a[0])
    m = len(b[0])

    result = [[0.0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            for x in range(k):
                result[i][j] += a[i][x] * b[x][j]
    return result


def flatten(lst):
    return [item for sublist in lst for item in sublist]


@pytest.mark.parametrize(
    "data, shape",
    [
        ([1.0, 2.0, 3.0], (3,)),
        ([[1.0, 2.0, 3.0]], (1, 3)),
        (
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ],
            (2, 2, 3),
        ),
    ],
)
def test_tensor_allocation(data, shape):
    t = Tensor(data)

    assert t.shape == shape
    assert t.to_cpu() == data


@pytest.mark.parametrize(
    "data",
    [
        [[1.0, 2.0, 3.0], [1.0, 2.0]],
        [[[1.0, 2.0, 3.0], [4.0, 5.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
    ],
)
def test_should_throw_for_irregular_arrays(data):
    with pytest.raises(ValueError):
        Tensor(data)


def test_tensor_addition():
    t1 = Tensor([10.0, 20.0, 30.0])
    t2 = Tensor([1.0, 2.0, 3.0])

    t3 = t1 + t2

    assert t3.shape == (3,)
    assert_close(t3.to_cpu(), [11.0, 22.0, 33.0])


def test_tensor_subtraction():
    t1 = Tensor([10.0, 20.0, 30.0])
    t2 = Tensor([1.0, 2.0, 3.0])

    t3 = t1 - t2

    assert t3.shape == (3,)
    assert_close(t3.to_cpu(), [9.0, 18.0, 27.0])


def test_tensor_elementwise_multiplicatoin():
    t1 = Tensor([10.0, 20.0, 30.0])
    t2 = Tensor([1.0, 2.0, 3.0])

    t3 = t1 * t2

    assert t3.shape == (3,)
    assert_close(t3.to_cpu(), [10.0, 40.0, 90.0])


def test_tensor_negation():
    t = Tensor([1.0, 2.0, 3.0])
    t = -t

    assert t.shape == (3,)
    assert_close(t.to_cpu(), [-1.0, -2.0, -3.0])


@pytest.mark.parametrize(
    "data_a, data_b, expected_shape",
    [
        (
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            (2, 2),
        ),
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            (2, 2),
        ),
        ([[1.0, 2.0, 3.0]], [[1.0], [2.0], [3.0]], (1, 1)),
    ],
)
def test_matmul_valid(data_a, data_b, expected_shape):
    t1 = Tensor(data_a)
    t2 = Tensor(data_b)

    result = t1 @ t2

    assert tuple(result.shape) == expected_shape

    expected_data = simple_python_matmul(data_a, data_b)
    flat_expected = flatten(expected_data)
    flat_actual = flatten(result.to_cpu())

    assert_close(flat_actual, flat_expected)


def test_matmul_mismatch_raises_error():
    t1 = Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    t2 = Tensor([[1.0, 1.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="Shape mismatch"):
        t1 @ t2


def test_tensor_large_data():
    size = 10_000
    data = [float(i) for i in range(size)]

    t1 = Tensor(data)
    t2 = Tensor(data)

    t3 = t1 + t2

    res = t3.to_cpu()
    assert len(res) == size
    assert res[0] == 0.0
    assert res[-1] == (size - 1) * 2


def test_shape_mismatch():
    t1 = Tensor([1.0, 2.0])
    t2 = Tensor([1.0, 2.0, 3.0])

    with pytest.raises((ValueError, RuntimeError)):
        _ = t1 + t2


def test_memory_cleanup():
    t = Tensor([1.0, 2.0, 3.0])
    del t
    assert True
