import numpy as np
import pytest

import ember as em


@pytest.fixture(autouse=True)
def _seed():
    """Seed NumPy's global generator and ember's before every test, so random
    inputs and initial weights are the same on every run."""
    np.random.seed(0)
    em.random.seed(0)
