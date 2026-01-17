import numpy as np

from ember import Tensor

a = np.random.uniform(0, 1, (10))
b = np.random.uniform(0, 1, (10, 10))

t_a = Tensor.from_np(a)
t_b = Tensor.from_np(b)

t_c = t_a + t_b
print(np.allclose(t_c.to_np(), a + b))
