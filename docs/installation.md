# Installation

Ember requires Python 3.11 or later. You can install it from source.

## Prerequisites

- **CMake** (>= 3.18)
- **C Compiler** (GCC, Clang, or MSVC)
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
