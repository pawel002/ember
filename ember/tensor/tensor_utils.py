def calculate_contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    ndim = len(shape)
    strides = [0] * ndim

    current_stride = 1
    for i in range(ndim - 1, -1, -1):
        strides[i] = current_stride
        current_stride *= shape[i]

    return tuple(strides)


def calculate_broadcast(
    shape_a: tuple[int, ...], shape_b: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """NumPy-style broadcast of two contiguous tensors: the output shape and,
    for each operand, the element stride along every output dimension (0 where
    the operand is broadcast)."""
    strides_a = calculate_contiguous_strides(shape_a)
    strides_b = calculate_contiguous_strides(shape_b)
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
