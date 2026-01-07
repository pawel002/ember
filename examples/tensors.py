from ember import Tensor

import numpy as np

from time import perf_counter

a = np.random.uniform(0, 1, (4096, 4096))
b = np.random.uniform(0, 1, (4096, 4096))

t_a = Tensor.from_np(a)
t_b = Tensor.from_np(b)

t = perf_counter()
c = a @ b
print(perf_counter() - t)

t = perf_counter()
t_c = t_a @ t_b
c = t_c.to_np()
print(perf_counter() - t)
