from ember import Tensor

import numpy as np


a = np.random.uniform(0, 1, (100, 50))
b = np.random.uniform(0, 1, (50, 200))

t_a = Tensor.from_np(a)
t_b = Tensor.from_np(b)

t_c = t_a @ t_b
print(t_c.shape)
