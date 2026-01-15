from typing import Any


def extract_data_info(
    data: Any,
) -> tuple[tuple[int, ...], type, list[int | float]]:
    """
    Analyzes a nested list to extract shape, inferred type, and flattened data.
    Returns:
        (shape, dtype, flat_list)
    """
    # 1. Handle Scalar Case (0-D Tensor)
    if not isinstance(data, list):
        if isinstance(data, bool):
            raise TypeError("Boolean data is not supported (ambiguous numeric type).")
        if not isinstance(data, (int, float)):
            raise TypeError(f"Unsupported data type: {type(data)}")
        return (), type(data), [data]

    # 2. Determine Expected Shape (The "Spine")
    # We trace data[0] -> data[0][0]... to determine what the shape SHOULD be.
    shape_agg = []
    temp = data
    while isinstance(temp, list):
        if len(temp) == 0:
            shape_agg.append(0)
            break
        shape_agg.append(len(temp))
        temp = temp[0]

    shape: tuple = tuple(shape_agg)

    # 3. Recursive Traversal (Flatten + Validate + Type Check)
    flat_data = []
    has_float = False

    def _recursive_scan(node: Any, depth: int):
        nonlocal has_float

        # CASE A: We expect a list at this depth (Internal Node)
        if depth < len(shape):
            if not isinstance(node, list):
                raise ValueError(
                    f"Ragged sequence: Expected list at depth {depth}, got scalar."
                )

            if len(node) != shape[depth]:
                raise ValueError(
                    f"Shape mismatch at depth {depth}. Expected length {shape[depth]}, got {len(node)}."
                )

            for item in node:
                _recursive_scan(item, depth + 1)

        # CASE B: We expect a scalar at this depth (Leaf Node)
        else:
            if isinstance(node, list):
                raise ValueError(
                    "Ragged sequence: Expected scalar, got list (too deep)."
                )

            if isinstance(node, bool):
                raise TypeError("Boolean data is not supported.")

            if isinstance(node, float):
                has_float = True
                flat_data.append(node)
            elif isinstance(node, int):
                flat_data.append(node)
            else:
                raise TypeError(f"Unsupported item type: {type(node)}")

    # Start scanning from depth 0
    _recursive_scan(data, 0)

    # 4. Final Type Promotion
    dtype = float if has_float else int

    if dtype == float and not all(isinstance(x, float) for x in flat_data):
        flat_data = [float(x) for x in flat_data]

    return shape, dtype, flat_data


def calculate_contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    ndim = len(shape)
    strides = [0] * ndim

    current_stride = 1
    for i in range(ndim - 1, -1, -1):
        strides[i] = current_stride
        current_stride *= shape[i]

    return tuple(strides)


def calculate_broadcast(
    shape_a: tuple[int, ...],
    strides_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    strides_b: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    ndim_a = len(shape_a)
    ndim_b = len(shape_b)
    ndim_out = max(ndim_a, ndim_b)

    padded_shape_a = (1,) * (ndim_out - ndim_a) + shape_a
    padded_shape_b = (1,) * (ndim_out - ndim_b) + shape_b

    padded_strides_a = (0,) * (ndim_out - ndim_a) + strides_a
    padded_strides_b = (0,) * (ndim_out - ndim_b) + strides_b

    out_shape = []
    out_strides_a = []
    out_strides_b = []

    for dim_a, str_a, dim_b, str_b in zip(
        padded_shape_a, padded_strides_a, padded_shape_b, padded_strides_b, strict=True
    ):
        if dim_a == dim_b:
            out_shape.append(dim_a)
            out_strides_a.append(str_a)
            out_strides_b.append(str_b)

        elif dim_a == 1:
            out_shape.append(dim_b)
            out_strides_a.append(0)
            out_strides_b.append(str_b)

        elif dim_b == 1:
            out_shape.append(dim_a)
            out_strides_a.append(str_a)
            out_strides_b.append(0)

        else:
            raise ValueError(
                f"Operands could not be broadcast together with shapes {shape_a} and {shape_b}"
            )

    return tuple(out_shape), tuple(out_strides_a), tuple(out_strides_b)
