from typing import Any, List, Tuple, Union


def extract_data_info(
    data: Any,
) -> Tuple[Tuple[int, ...], type, List[Union[int, float]]]:
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

    shape: Tuple = tuple(shape_agg)

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
