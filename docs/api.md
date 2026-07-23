# API Reference

## Class `ember.Tensor`

The core data structure of the Ember library. A `Tensor` represents a multi-dimensional array of elements, similar to NumPy arrays but optimized for Ember's backend (C/CUDA).

### Constructor

#### `__init__(data)`
Creates a `Tensor` from a Python list, a scalar, or a NumPy array.

- **Arguments:**
  - `data` (list | number | numpy.ndarray): The data to initialize the tensor with. NumPy arrays are cast to `float32`.
- **Returns:** A new `Tensor` instance.

```python
import ember
import numpy as np

t = ember.Tensor([1.0, 2.0, 3.0])
u = ember.Tensor(np.eye(3, dtype=np.float32))
```

> **Note:** the backend is currently `float32`-only. Integer lists are stored, but all computation happens in `float32`.

#### `from_np(array)`
Creates a `Tensor` from a NumPy array.

- **Arguments:**
  - `array` (numpy.ndarray): The NumPy array to convert.
- **Returns:** A new `Tensor` instance.

### Methods

#### `to_np() -> numpy.ndarray`
Converts the tensor back to a NumPy array.

#### `to_list() -> list`
Converts the tensor data to a (possibly nested) Python list.

#### `reshape(new_shape: tuple[int, ...]) -> Tensor`
Reshapes the tensor **in place** and returns it (the same object).

- **Arguments:**
  - `new_shape`: The new shape. One dimension can be -1, which will be inferred.

### Operators

`Tensor` objects support standard Python operators.

| Operator | Method | Description | Formula |
| :--- | :--- | :--- | :--- |
| `+` | `__add__` | Elementwise addition | $c_i = a_i + b_i$ |
| `-` | `__sub__` | Elementwise subtraction | $c_i = a_i - b_i$ |
| `*` | `__mul__` | Elementwise multiplication | $c_i = a_i \cdot b_i$ |
| `/` | `__truediv__` | Elementwise division | $c_i = \frac{a_i}{b_i}$ |
| `@` | `__matmul__` | Matrix multiplication (2-D only) | $C_{ij} = \sum_k A_{ik} B_{kj}$ |
| `**` | `__pow__` | Elementwise power | $c_i = a_i^{b_i}$ |
| `-` | `__neg__` | Elementwise negation | $b_i = -a_i$ |
| `>` | `__gt__` | Elementwise greater than | $c_i = 1 \text{ if } a_i > b_i \text{ else } 0$ |
| `<` | `__lt__` | Elementwise less than | $c_i = 1 \text{ if } a_i < b_i \text{ else } 0$ |

The in-place forms `+=`, `-=`, `*=`, `/=` are also supported and mutate the tensor
object in place (optimizers rely on this). Scalar operands and NumPy-style
broadcasting (for `+`, `-`, `*`, `/`) are supported throughout.

---

## Functional API

All functional operations are available directly under the `ember` namespace.

### Elementwise Math Operations

Functions that operate on each element of the input tensor independently.
For an input tensor $A$ with elements $a_i$, the output tensor $B$ has elements $b_i$:

$$b_i = f(a_i)$$

#### Trigonometric

| Function | Description | Equation |
| :--- | :--- | :--- |
| `ember.sin(x)` | Sine | $b_i = \sin(a_i)$ |
| `ember.cos(x)` | Cosine | $b_i = \cos(a_i)$ |
| `ember.tan(x)` | Tangent | $b_i = \tan(a_i)$ |
| `ember.ctg(x)` | Cotangent | $b_i = \cot(a_i)$ |

#### Hyperbolic

| Function | Description | Equation |
| :--- | :--- | :--- |
| `ember.sinh(x)` | Hyperbolic Sine | $b_i = \sinh(a_i)$ |
| `ember.cosh(x)` | Hyperbolic Cosine | $b_i = \cosh(a_i)$ |
| `ember.tanh(x)` | Hyperbolic Tangent | $b_i = \tanh(a_i)$ |
| `ember.ctgh(x)` | Hyperbolic Cotangent | $b_i = \coth(a_i)$ |

#### Other

| Function | Description | Equation |
| :--- | :--- | :--- |
| `ember.exp(x)` | Exponential | $b_i = e^{a_i}$ |
| `ember.sqrt(x)` | Square root | $b_i = \sqrt{a_i}$ |

### Reductions & Shape

| Function | Description |
| :--- | :--- |
| `ember.sum(x)` | Sum of all elements (returns a `float`). |
| `ember.sum(x, axis=k)` | Sum along `axis` `k`, returning a `Tensor`. |
| `ember.T(x)` | Transpose of a 2-D tensor. |

### Binary Elementwise Operations

Functions that extend elementwise operations to two tensors (or a tensor and a scalar).
For inputs $A, B$, the output $C$ is:

$$c_i = f(a_i, b_i)$$

| Function | Description | Equation |
| :--- | :--- | :--- |
| `ember.max(a, b)` | Elementwise maximum | $c_i = \max(a_i, b_i)$ |
| `ember.min(a, b)` | Elementwise minimum | $c_i = \min(a_i, b_i)$ |

---

## Random Module (`ember.random`)

Utilities for generating tensors with random or constant values.

#### `ember.random.seed(value)`
Seeds the random number generator so that `uniform` (and any layer that uses it,
such as weight initialisation and dropout) becomes reproducible.

- **Arguments:**
  - `value` (int): The seed.

#### `ember.random.uniform(low, high, size)`
Draws samples from a uniform distribution.

- **Arguments:**
  - `low` (float): Lower boundary of the output interval.
  - `high` (float): Upper boundary of the output interval.
  - `size` (tuple[int]): Shape of the output tensor.
- **Equation:** $x_i \sim U(\text{low}, \text{high})$

#### `ember.random.constant(value, size)`
Return a new tensor of given shape and type, filled with `value`.

- **Arguments:**
  - `value` (float): Fill value.
  - `size` (tuple[int]): Shape of the output tensor.
- **Equation:** $x_i = \text{value}$

#### `ember.random.zeros(size)`
Returns a new tensor of given shape, filled with zeros.

#### `ember.random.ones(size)`
Returns a new tensor of given shape, filled with ones.

---

## Neural Networks (`ember.nn`)

Components for building neural networks. All public classes are available
directly under `ember.nn`, e.g. `ember.nn.Linear`, `ember.nn.ReLU`.

Every layer implements a common interface (defined by `ember.nn.base.Layer`):

- `forward(x, training) -> Tensor` and `backward(grad_y) -> Tensor`
- `parameters() -> list[Tensor]` and `gradients() -> list[Tensor | None]`
- `reset()` to (re-)initialise state
- calling the layer (`layer(x)`) runs `forward` and caches inputs/outputs.

### Layers

#### `ember.nn.Linear(in_features, out_features)`
Applies a linear transformation to the incoming data.

- **Arguments:**
  - `in_features` (int): Size of each input sample.
  - `out_features` (int): Size of each output sample.
- **Forward:** $y = x W + b$
  - Where $W$ is weight matrix of shape `(in_features, out_features)` and $b$ is bias of shape `(out_features,)`.

**Methods:**

- `reset()`: Re-initializes weights and biases.
- `parameters() -> list[Tensor]`: Returns `[w, b]`.
- `gradients() -> list[Tensor | None]`: Returns `[grad_w, grad_b]`.

#### `ember.nn.Dropout(p)`
Inverted dropout. During training, zeroes each element with probability `p` and
scales the survivors by $1/(1-p)$; during inference it is the identity.

- **Arguments:**
  - `p` (float): Drop probability, `0 <= p < 1`.

#### `ember.nn.Sequential(*layers)`
Chains layers together, running them in order on `forward` and in reverse on
`backward`. `parameters()`/`gradients()` aggregate across all child layers.

```python
import ember.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

### Activations

Activation functions applied elementwise.

#### `ember.nn.activations.ReLU`
Rectified Linear Unit.

- **Equation:** $f(x) = \max(0, x)$

#### `ember.nn.activations.Sigmoid`
Sigmoid function.

- **Equation:** $f(x) = \frac{1}{1 + e^{-x}}$

#### `ember.nn.activations.Tanh`
Hyperbolic Tangent function.

- **Equation:** $f(x) = \tanh(x)$

#### `ember.nn.activations.GELU`
Gaussian Error Linear Unit (using `tanh` approximation).

- **Equation:** $f(x) \approx 0.5 x (1 + \tanh(0.8 x))$

---

## Optimizers (`ember.optim`)

Optimizers update a list of parameters from a matching list of gradients. All
optimizers implement `ember.optim.base.Optimizer`:

- `__init__(parameters, ...)`
- `apply(gradients: list[Tensor]) -> None`

#### `ember.optim.Adam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)`
Adam optimizer.

- **Arguments:**
  - `parameters` (list[Tensor]): Parameters to optimize.
  - `lr` (float): Learning rate.
  - `betas` (tuple[float, float]): Coefficients for the running averages of the gradient and its square.
  - `eps` (float): Term added to the denominator for numerical stability.

```python
import ember.optim as optim

opt = optim.Adam(model.parameters(), lr=1e-3)
# ... after a forward/backward pass ...
opt.apply(model.gradients())
```
