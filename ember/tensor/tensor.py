from typing import List, Literal
from functools import reduce

import math

from ember._core import _Tensor

Types = Literal["int32", "float32"]

class Tensor:
    dtype: Types
    shape: List[int]
    _core: _Tensor

    def __init__(self, data, _core=None):
        if _core is not None:
            self._core = _core
            self.shape = (self._core.size,)
            return

        if isinstance(data, list):
            size = len(data)
            self._core = _Tensor(size) 
            self._core.copy_from_list(data)
            self.shape = (size,)
        else:
            raise ValueError("Unsupported data type")

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Can only add Tensors")
            
        result_core = self._core._add(other._core)
        return Tensor(data=None, _core=result_core)

    def to_cpu(self):
        return self._core.to_list()
    
    def reshape(self, new_shape: List[int]) -> Tensor:
        """
        
        """
        total_elements = math.prod(self.shape)

        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError("Only one dimension can be -1 (inferred)")
            
            known_prod = -1 * math.prod(new_shape)
            if total_elements % known_prod != 0:
                raise ValueError(f"Cannot reshape size {total_elements} into {new_shape}")
            
            inferred_dim = total_elements // known_prod
            new_shape = [x if x != -1 else inferred_dim for x in new_shape]

        elif math.prod(new_shape) != total_elements:
                raise ValueError(f"Cannot reshape size {total_elements} into {new_shape}")
        
        self.shape = new_shape
        return self
        
    def __repr__(self):
        return f"Tensor({self.to_cpu()})"