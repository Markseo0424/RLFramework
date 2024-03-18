from typing import overload, Self

import numpy as np
import torch
from .wrapper import Wrapper
from .tensor import Tensor


class Array(Wrapper):
    def __init__(self, *args, data: dict = None, **kwargs):
        super().__init__(data)

        self.__ndarray = np.array(*args, **kwargs)

    def __getitem__(self, item):
        if self.data is not None:
            slice_data = {}
            for key in self.data.keys():
                if self.data[key].shape[0] == self.__ndarray.shape[0]:
                    slice_data[key] = self.data[key][item]
                else:
                    slice_data[key] = self.data[key]
        else:
            slice_data = None

        return Array(self.__ndarray[item], data=slice_data)

    def __getattr__(self, item):
        return getattr(self.__ndarray, item)

    def __mul__(self, other):
        if isinstance(other, Wrapper):
            return self * other.item()
        elif isinstance(other, torch.Tensor):
            return Tensor(torch.tensor(self.__ndarray).to(other) * other)
        else:
            return Array(self.item() * other)

    def __add__(self, other):
        if isinstance(other, Wrapper):
            return self + other.item()
        elif isinstance(other, torch.Tensor):
            return Tensor(torch.tensor(self.__ndarray).to(other) + other)
        else:
            return Array(self.item() + other)

    def __matmul__(self, other):
        if isinstance(other, Wrapper):
            return self @ other.item()
        elif isinstance(other, torch.Tensor):
            return Tensor(torch.tensor(self.__ndarray).to(other) @ other)
        else:
            return Array(self.item() @ other)

    def __sub__(self, other):
        if isinstance(other, Wrapper):
            return self - other.item()
        elif isinstance(other, torch.Tensor):
            return Tensor(torch.tensor(self.__ndarray).to(other) - other)
        else:
            return Array(self.item() - other)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __rmatmul__(self, other):
        if isinstance(other, Wrapper):
            return other.item() @ self
        elif isinstance(other, torch.Tensor):
            return  Tensor(other @ torch.tensor(self.__ndarray).to(other))
        else:
            return Array(other @ self.item())

    def __rsub__(self, other):
        if isinstance(other, Wrapper):
            return other.item() - self
        elif isinstance(other, torch.Tensor):
            return Tensor(other - torch.tensor(self.__ndarray).to(other))
        else:
            return Array(other - self.item())

    def reshape(self, *args, **kwargs):
        return Array(self.__ndarray.reshape(*args, **kwargs))

    def item(self):
        return self.__ndarray
