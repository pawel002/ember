# Ember

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey)

![](figures/ember.png)

**Ember** is a high-performance, lightweight deep learning library for Python. It provides a familiar NumPy-like interface with heavy lifting done by optimized C (CPU) and CUDA (GPU) backends.

## Key Features

- **Tensor Operations**: Fast element-wise operations and matrix multiplication.
- **Hardware Acceleration**: Automatic CUDA support on NVIDIA GPUs.
- **Pythonic**: Seamless integration with NumPy and Python lists.
- **Lightweight**: Minimal dependencies, easy to install.

## Installation

You can download and install it in a UV virtual environment using commands:

```bash
git clone https://github.com/pawel002/ember
cd ember
uv sync
source .venv/bin/activate
uv pip install -e .
```

## Quick Start

```python
from ember import Tensor

# Create tensors
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])

# Perform operations
c = a * b
print(c.to_list()) # Output: [4.0, 10.0, 18.0]
print(c.to_np())   # Output: np.array([4.0, 10.0, 18.0])
```

## Documentation

Full documentation is available in the `docs/` directory or can be built locally using MkDocs (by switching to docs dependency group):

```bash
uv sync --group docs
uv run mkdocs serve
```

## License

This project is licensed under the terms of the MIT license.
