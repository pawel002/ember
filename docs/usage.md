# Usage

Ember provides a `Tensor` class and various mathematical operations.

## Creating Tensors

You can create tensors from Python lists or NumPy arrays.

```python
import ember as em
import numpy as np

from ember import Tensor

# From list
a = Tensor([1, 2, 3])

# From NumPy
np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = Tensor(np_arr) # Automatically converts from buffer
```

## Operations

Ember supports element-wise operations and broadcasting with scalars.

### Arithmetic

```python
c = a + b
d = a * 2.0
e = em.max(a, b)
```

### Trigonometric & Hyperbolic

```python
# Standard trig
s = em.sin(a)
c = em.cos(a)

# Hyperbolic
sh = em.sinh(a)
th = em.tanh(a)
```

### Matrix Multiplication

```python
# Create 2x2 matrices (flattened logic for now, or reshape support pending)
# Note: Current implementation is basic.
pass
```

## Data Exchange

Convert back to standard Python/NumPy types easily.

```python
# To List
data_list = c.to_list()

# To NumPy
data_np = c.to_np()
```
