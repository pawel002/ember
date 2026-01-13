# API Reference

## Class `ember.Tensor`

The core data structure of the Ember library. A `Tensor` represents a multi-dimensional array of elements, similar to NumPy arrays but optimized for Ember's backend (C/CUDA).

### Constructor

#### `__init__(data)`
Creates a `Tensor` from a Python list or compatible data structure.

- **Arguments:**
  - `data` (list | number): The data to initialize the tensor with.
- **Returns:** A new `Tensor` instance.

```python
import ember
t = ember.Tensor([1.0, 2.0, 3.0])
```

#### `from_np(array)`
Creates a `Tensor` from a NumPy array.

- **Arguments:**
  - `array` (numpy.ndarray): The NumPy array to convert.
- **Returns:** A new `Tensor` instance.

### Methods

#### `to_np() -> numpy.ndarray`
Converts the tensor back to a NumPy array.

#### `to_cpu() -> list`
Converts the tensor data to a Python list.

#### `reshape(new_shape: tuple[int, ...]) -> Tensor`
Returns a `Tensor` with the same data but a different shape.

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
| `@` | `__matmul__` | Matrix multiplication | $C_{ij} = \sum_k A_{ik} B_{kj}$ |
| `-` | `__neg__` | Elementwise negation | $b_i = -a_i$ |
| `>` | `__gt__` | Elementwise greater than | $c_i = 1 \text{ if } a_i > b_i \text{ else } 0$ |

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

Components for building neural networks.
*Note: You may need to import submodules explicitly, e.g., `from ember.nn.layers import Linear`.*

### Layers

#### `ember.nn.layers.Linear(in_features, out_features)`
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
