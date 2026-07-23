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

Use the `@` operator for 2-D matrix multiplication.

```python
import ember as em
from ember import Tensor

a = Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
b = Tensor([[5.0, 6.0], [7.0, 8.0]])  # (2, 2)

c = a @ b            # matrix product
d = em.T(a)          # transpose
s = em.sum(a)        # sum of all elements -> float
row_sums = em.sum(a, axis=1)  # sum along an axis -> Tensor
```

## Data Exchange

Convert back to standard Python/NumPy types easily.

```python
# To a (nested) Python list
data_list = c.to_list()

# To NumPy
data_np = c.to_np()
```

## Building and Training a Model

Ember ships small `nn` and `optim` modules with explicit (manual) gradients.

```python
import numpy as np

import ember as em
import ember.loss as loss
import ember.nn as nn
import ember.optim as optim
from ember import Tensor

em.random.seed(0)  # reproducible weight init

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)
opt = optim.Adam(model.parameters(), lr=1e-2)
criterion = loss.MSELoss()

x = Tensor(np.random.randn(16, 4).astype(np.float32))
target = Tensor(np.random.randn(16, 1).astype(np.float32))

# forward + loss
pred = model(x, training=True)
value = criterion(pred, target)

# backward, then an optimizer step
model.backward(criterion.backward())
opt.apply(model.gradients())
```
