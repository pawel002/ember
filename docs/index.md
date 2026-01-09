# Ember

**Ember** is a lightweight deep learning Python package with C and CUDA implementations, designed for performance and simplicity.

## Features

- **Lightweight**: Minimal overhead, focused on core tensor operations.
- **Hardware Accelerated**:
  - Pure C implementation for CPU.
  - CUDA implementation for NVIDIA GPUs.
- **NumPy Compatible**: Seamless conversion to and from NumPy arrays.
- **Autograd Support**: (Upcoming) Automatic differentiation for training neural networks.

## Quick Start

```python
import ember
import numpy as np

# Create a tensor from a list
t = ember.Tensor([1.0, 2.0, 3.0])

# Perform operations
t2 = ember.sin(t)
print(t2.to_list())
# Output: [0.84147096, 0.90929741, 0.14112000] (approx)
```
