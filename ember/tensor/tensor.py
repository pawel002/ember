from . import _core 

class Tensor:
    def __init__(self, data, _core=None):
        if _core is not None:
            self._core = _core
            self.shape = (self._core.size,)
            return

        if isinstance(data, list):
            size = len(data)
            self._core = _core._CudaTensor(size) 
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

    def __repr__(self):
        return f"Tensor({self.to_cpu()})"