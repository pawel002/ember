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

> The CUDA backend is compiled automatically when a CUDA toolkit is detected;
> otherwise Ember falls back to the pure-C CPU backend. The rest of the library
> is identical either way.

## Quick Start

```python
from ember import Tensor

# Create tensors (from lists or NumPy arrays)
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])

# Perform operations
c = a * b
print(c.to_list()) # Output: [4.0, 10.0, 18.0]
print(c.to_np())   # Output: np.array([4.0, 10.0, 18.0])
```

Ember also ships small `nn` and `optim` modules:

```python
import ember as em
import ember.nn as nn
import ember.optim as optim

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
opt = optim.Adam(model.parameters(), lr=1e-2)

y = model(x, training=True)   # x is a Tensor of shape (batch, 4)
model.backward(grad)          # your loss gradient w.r.t. y
opt.apply(model.gradients())
```

## Extending Ember

Adding a new element-wise operator is a one-line change to a single source of
truth (`src/tensor/operators.def`), which generates the C declaration, the CPU
kernel, the CUDA kernel, and the Python binding. See the
[Extending Ember](https://pawel002.github.io/ember/extending/) guide.

## Documentation

Full documentation is available under [this link](https://pawel002.github.io/ember/) or can be built locally using MkDocs (by switching to docs dependency group):

```bash
uv sync --group docs
uv run mkdocs serve
```

## License

This project is licensed under the terms of the MIT license.
