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

#### `reshape(shape: Sequence[int]) -> Tensor`
Returns the tensor's data with a new shape. The original tensor keeps its shape.

No data is copied: tensors are always contiguous, so the result shares the
original's buffer, and an in-place write through either one is visible in both.

- **Arguments:**
  - `shape`: The new shape. One dimension can be -1, which will be inferred.

### Operators

`Tensor` objects support standard Python operators.

| Operator | Method | Description | Formula |
| :--- | :--- | :--- | :--- |
| `+` | `__add__` | Elementwise addition | $c_i = a_i + b_i$ |
| `-` | `__sub__` | Elementwise subtraction | $c_i = a_i - b_i$ |
| `*` | `__mul__` | Elementwise multiplication | $c_i = a_i \cdot b_i$ |
| `/` | `__truediv__` | Elementwise division | $c_i = \frac{a_i}{b_i}$ |
| `@` | `__matmul__` | Matrix multiplication (2-D, or batched N-D with matching batch dims) | $C_{ij} = \sum_k A_{ik} B_{kj}$ |
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
| `ember.log(x)` | Natural logarithm | $b_i = \ln(a_i)$ |
| `ember.sqrt(x)` | Square root | $b_i = \sqrt{a_i}$ |

### Reductions & Shape

| Function | Description |
| :--- | :--- |
| `ember.sum(x)` | Sum of all elements (returns a `float`). |
| `ember.sum(x, axis=k, keepdims=False)` | Sum along `axis` `k`, returning a `Tensor`. |
| `ember.amax(x, axis=k, keepdims=False)` | Maximum along `axis` `k` (reduction; distinct from the element-wise `ember.max`). |
| `ember.softmax(x, axis=-1)` | Numerically-stable softmax along `axis`. |
| `ember.T(x)` | Transpose of a 2-D tensor. |

Matrix multiplication (`@`) supports 2-D operands and batched N-D operands
whose leading (batch) dimensions match, e.g. `(B, n, k) @ (B, k, m) -> (B, n, m)`.

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

Random tensors.

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

`ember.random.constant(value, size)`, `ember.random.zeros(size)` and
`ember.random.ones(size)` are kept as aliases of the creation functions below.

## Creation

#### `ember.full(shape, fill_value)`
A new tensor of the given shape with every element set to `fill_value`.

#### `ember.zeros(shape)` / `ember.ones(shape)`
A new tensor of the given shape filled with zeros / ones.

---

## Neural Networks (`ember.nn`)

Components for building neural networks. All public classes are available
directly under `ember.nn`, e.g. `ember.nn.Linear`, `ember.nn.ReLU`.

Every layer implements a common interface (defined by `ember.nn.base.Layer`):

- `forward(x, training) -> Tensor` and `backward(grad_y) -> Tensor`; calling
  the layer (`layer(x, training=True)`) runs `forward`, which caches what
  `backward` needs.
- `parameters() -> list[Tensor]` and `gradients() -> list[Tensor | None]`, in
  the same fixed order.
- `sublayers()`: the child layers of a composite layer. A layer built from other
  layers only overrides this, and `parameters()`/`gradients()` concatenate its
  children's.

Parameters are created when the layer is constructed.

### Layers

#### `ember.nn.Linear(in_features, out_features)`
Applies a linear transformation to the incoming data.

- **Arguments:**
  - `in_features` (int): Size of each input sample.
  - `out_features` (int): Size of each output sample.
- **Forward:** $y = x W + b$
  - Where $W$ is weight matrix of shape `(in_features, out_features)` and $b$ is bias of shape `(out_features,)`.
  - `x` has shape `(..., in_features)`: any leading dimensions (e.g. `(batch, seq)`) are kept, and the output has shape `(..., out_features)`.

**Methods:**

- `parameters() -> list[Tensor]`: Returns `[weight, bias]`.
- `gradients() -> list[Tensor | None]`: Returns `[grad_weight, grad_bias]`.

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

#### `ember.nn.ReLU`
Rectified Linear Unit.

- **Equation:** $f(x) = \max(0, x)$

#### `ember.nn.Sigmoid`
Sigmoid function.

- **Equation:** $f(x) = \frac{1}{1 + e^{-x}}$

#### `ember.nn.Tanh`
Hyperbolic Tangent function.

- **Equation:** $f(x) = \tanh(x)$

#### `ember.nn.GELU`
Gaussian Error Linear Unit (using `tanh` approximation).

- **Equation:** $f(x) \approx 0.5 x (1 + \tanh(0.8 x))$

### Transformer layers

Building blocks for attention models. Each is backed by fused CUDA kernels
(LayerNorm is a single kernel; attention is a fused flash-attention kernel that
never materializes the score matrix). They share the `Layer` interface, so they
compose with `Sequential`, the optimizers and the losses like any other layer.

#### `ember.nn.LayerNorm(dim, eps=1e-5)`
Layer normalization over the last dimension, with learned scale/shift:
$y = \gamma \, (x - \mu) / \sqrt{\sigma^2 + \epsilon} + \beta$. Accepts any shape
whose last axis is `dim`. `parameters()` returns `[gamma, beta]`.

#### `ember.nn.Embedding(num_embeddings, embedding_dim)`
A lookup table. `forward(idx)` takes an integer index array (NumPy array or
nested list, any shape) and returns `(*idx.shape, embedding_dim)`. `backward`
scatter-adds gradients into the used rows and returns `None` (integer inputs
have no gradient); `parameters()` returns `[weight]`.

#### `ember.nn.MultiHeadAttention(embed_dim, num_heads, causal=False)`
Multi-head self-attention on `(batch, seq, embed_dim)` inputs:
$\mathrm{softmax}(QK^\top / \sqrt{d_h})V$ per head. With `causal=True` each query
attends only to keys at or before its position (decoder / GPT-style masking).
The four projections are `Linear` layers; `parameters()` aggregates all of them.

The attention itself runs as a single fused ("flash") kernel when the backend
has one for this `head_dim` (16, 32, 64 or 128 -- check the `fused` attribute):
scores are tiled through shared memory and never written to global memory, and
backward recomputes them from the saved log-sum-exp. Other head dimensions fall
back automatically to the composed batched-GEMM + fused-softmax path, which
materializes the `(seq, seq)` score matrix. Both paths compute the same thing.

#### `ember.nn.FeedForward(dim, hidden=None, activation="gelu")`
Position-wise MLP: `Linear(dim, hidden) -> activation -> Linear(hidden, dim)`.
`hidden` defaults to `4 * dim`. `activation` is `"gelu"` or `"relu"`.

#### `ember.nn.PositionalEncoding(dim, max_len=5000)`
Fixed sinusoidal positional encoding added to a `(batch, seq, dim)` input. No
learnable parameters.

#### `ember.nn.TransformerEncoderLayer(embed_dim, num_heads, ff_hidden=None, causal=False, activation="gelu")`
One pre-norm block: `h = x + Attn(LayerNorm(x))`, then
`out = h + FeedForward(LayerNorm(h))`.

#### `ember.nn.TransformerEncoder(num_layers, embed_dim, num_heads, ff_hidden=None, causal=False, activation="gelu", final_norm=True)`
A stack of `TransformerEncoderLayer` blocks with an optional final `LayerNorm`.

```python
import ember.nn as nn

encoder = nn.TransformerEncoder(num_layers=4, embed_dim=128, num_heads=8, causal=True)
y = encoder(x, training=True)  # x: (batch, seq, 128)
dx = encoder.backward(grad_y)
```

---

## Optimizers (`ember.optim`)

Optimizers update a list of parameters from a matching list of gradients. All
optimizers implement `ember.optim.base.Optimizer`:

- `__init__(parameters, ...)`
- `apply(gradients: list[Tensor]) -> None`
- `state_dict() -> dict` / `load_state_dict(state)`: snapshot and restore the
  optimizer's state (momentum / moment buffers and the step counter) as NumPy
  arrays. Hyperparameters such as `lr` are not part of the state. Usually you
  call `ember.save` / `ember.load` instead (see [Checkpoints](#checkpoints)).

#### `ember.optim.SGD(parameters, lr=0.01, momentum=0.0)`
Stochastic gradient descent with optional (Polyak) momentum: `v = momentum*v + g`,
then `p -= lr*v`. With `momentum=0` this is plain gradient descent.

#### `ember.optim.Adam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, capturable=False, foreach=False)`
Adam optimizer.

- **Arguments:**
  - `parameters` (list[Tensor]): Parameters to optimize.
  - `lr` (float): Learning rate.
  - `betas` (tuple[float, float]): Coefficients for the running averages of the gradient and its square.
  - `eps` (float): Term added to the denominator for numerical stability.
  - `capturable` (bool): Keep the step counter and bias corrections on-device so
    the update stays exact under CUDA-graph capture.
  - `foreach` (bool): Update every parameter in one grouped kernel launch
    instead of one launch per parameter. Cannot be combined with `capturable`.

#### `ember.optim.AdamW(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, capturable=False, foreach=False)`
Adam with decoupled weight decay (`p *= 1 - lr*weight_decay` applied directly to
the parameters rather than folded into the gradient). Takes the same
`capturable` / `foreach` options as `Adam`.

```python
import ember.optim as optim

opt = optim.Adam(model.parameters(), lr=1e-3)
# ... after a forward/backward pass ...
opt.apply(model.gradients())
```

---

## Checkpoints

#### `ember.save(path, model=None, optimizer=None, **metadata)`
Writes the model's parameters and/or the optimizer's state to `path`.

- **Arguments:**
  - `path` (str | PathLike): Output file. It is written to exactly this path,
    with no `.npz` suffix added.
  - `model`: Anything with a `parameters()` method (any `nn.Layer`, or your own
    model class).
  - `optimizer` (Optimizer): An `ember.optim` optimizer.
  - `**metadata`: Extra values to store with the checkpoint (step, epoch,
    config, ...). They must be JSON-serializable.

The save goes to a temporary file that is then renamed over `path`, so an
interrupted save never leaves a half-written checkpoint.

#### `ember.load(path, model=None, optimizer=None) -> dict`
Restores `model` and/or `optimizer` from a checkpoint and returns the saved
`**metadata`.

Parameters are matched by their position in `model.parameters()`, so build the
model with the same architecture first. Values are copied **in place** into the
existing tensors. An optimizer built over `model.parameters()`, or a step
captured with `ember.cuda.capture`, keeps working without being rebuilt. All
shapes are checked before anything is written: a mismatch raises `ValueError`
and leaves the model and optimizer unchanged.

The format is a plain `.npz` archive: float32 arrays plus a JSON header. It is
read with `allow_pickle=False`, so loading a checkpoint never runs code from it.
You can also inspect a checkpoint with `numpy.load`.

```python
import ember

# save during training
ember.save("ckpt.npz", model, opt, step=step)

# resume: build the same model + optimizer, then load into them
model = build_model()
opt = ember.optim.AdamW(model.parameters(), lr=1e-3)
step = ember.load("ckpt.npz", model, opt)["step"]

# inference only
ember.load("ckpt.npz", model)
```

---

## Losses (`ember.loss`)

Losses implement `ember.loss.base.Loss`:

- `forward(pred, target) -> float` (also callable) computes the scalar loss and
  caches what `backward` needs; `backward() -> Tensor` then returns the
  gradient of the loss with respect to `pred`.
- `gradient(pred, target) -> Tensor` returns that gradient directly, without
  reading the scalar loss back to the host. Use it in hot loops and in a step
  captured with `ember.cuda.capture`, where a host read cannot happen.

#### `ember.loss.MSELoss()`
Mean squared error over all elements: `mean((pred - target) ** 2)`.

#### `ember.loss.CrossEntropyLoss()`
Softmax cross-entropy for classification, averaged over the batch. Expects
`pred` to be raw logits of shape `(batch, classes)` and `target` to be one-hot
probabilities of the same shape. Uses the log-sum-exp trick for stability.

```python
import ember.loss as loss

criterion = loss.CrossEntropyLoss()
value = criterion(logits, one_hot_targets)  # scalar float
grad = criterion.backward()  # Tensor, dL/dlogits
```

---

## Device control (`ember.cuda`)

#### `ember.cuda.is_available()`
Whether ember was built with the CUDA backend.

#### `ember.cuda.sync()`
Block until all queued GPU work has finished.

#### `ember.cuda.empty_cache()`
Release ember's cached device-memory pool back to the driver.

#### `ember.cuda.memory_allocated() -> int`
Bytes of device memory held by live tensors. Freed blocks that ember keeps
cached for reuse are not counted. Always 0 on the CPU backend.

#### `ember.cuda.set_matmul_tf32(enabled)` / `ember.cuda.get_matmul_tf32()`
Run cuBLAS GEMMs on the TF32 tensor cores instead of the fp32 pipeline —
roughly 3x the GEMM throughput on Ampere and later, at a 10-bit input mantissa
with fp32 accumulation. The same trade as PyTorch's
`torch.backends.cuda.matmul.allow_tf32`, and **off by default** for the same
reason: it is a real precision change.

#### `ember.cuda.capture(step, warmup=3)`
Record a fixed-shape training step into a replayable `Graph`. See
[Performance](performance.md) for the requirements. If `step` raises during
capture, the capture is abandoned and the exception propagates.

On the CPU backend these are no-ops (`set_matmul_tf32` stores the flag but
nothing reads it).
