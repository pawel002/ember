from __future__ import annotations
from typing import List, Literal, Any, Dict

import math

from ember._core import _Tensor
from .tensor_utils import extract_data_info

Types = Literal["int32", "float32"]
_Types_lookup: Dict[type, Types] = {
    int: "int32",
    float: "float32"
}

class Tensor:
    dtype: Types
    shape: List[int]
    _core: _Tensor

    def __init__(self, data: Any):
        shape, dtype_cls, flat_data = extract_data_info(data)
        self.shape = shape
        self.dtype = _Types_lookup.get(dtype_cls, "float32")
        self._core = _Tensor(math.prod(shape))
        self._core.copy_from_list(flat_data)

    @classmethod
    def _from_core(cls, core: _Tensor, shape: List[int], dtype: Types) -> Tensor:
        obj = cls.__new__(cls)
        
        obj._core = core
        obj.shape = shape
        obj.dtype = dtype
        
        return obj

    def __add__(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type(s) for +: Tensor and '{type(other).__name__}'")
            
        result_core = self._core._add(other._core)
        result_shape = self.shape 
        result_dtype = self.dtype 

        return Tensor._from_core(result_core, result_shape, result_dtype)
    
    def __sub__(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type(s) for -: Tensor and '{type(other).__name__}'")
            
        result_core = self._core._subtract(other._core)
        result_shape = self.shape
        result_dtype = self.dtype

        return Tensor._from_core(result_core, result_shape, result_dtype)

    def to_cpu(self) -> List[Types]:
        return self._core.to_list(self.shape)
    
    def reshape(self, new_shape: List[int]) -> Tensor:
        """
        Modifies the shape of the vector in place, doesn't actually move memory.
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

    def _set_core(self, core: _Tensor) -> None:
        """
        This privates funtions in place sets the underlying _Tensor core
        for public Tensor wrapper class.
        """

        if core is None:
            raise ValueError("Core cannot be None")
        
        if not isinstance(core, _Tensor):
            raise TypeError("Core needs to be of type _Tensor")
        
        self._core = core

    def __repr__(self):
        return f"Tensor({self.to_cpu()})"