# Installation

Ember requires Python 3.11 or later. You can install it from source.

## Prerequisites

- **CMake** (>= 3.24; `uv`/`pip` fetch a recent one automatically)
- **C Compiler** (GCC or Clang; CI builds with GCC on Linux)
- **CUDA Toolkit** (Optional, for GPU support)

## Installing from Source

1. Clone the repository:

   ```bash
   git clone https://github.com/pawel002/ember
   cd ember
   ```

2. Create a virtual environment (recommended) and download dependencies:

   ```bash
   uv sync
   source .venv/bin/activate
   ```

3. Install the package:

   ```bash
   uv pip install -e .
   ```

   The install is editable: Python changes take effect immediately, and
   changes to the C/CUDA sources need the same command again.

## Build options

Pass CMake options with `-C cmake.define.<NAME>=<VALUE>`:

- `EMBER_CUDA=OFF` builds the CPU backend even when a CUDA compiler is found.
- `CMAKE_CUDA_ARCHITECTURES=<list>` sets the GPU architectures to compile for
  (default: the GPU in the build machine, e.g. `89` for an RTX 6000 Ada).

```bash
uv pip install -e . -C cmake.define.EMBER_CUDA=OFF
```
