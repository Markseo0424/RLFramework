from typing import overload, Self

import numpy as np
import torch
from .wrapper import Wrapper


class Tensor(Wrapper):
    def __init__(self, *args, data: dict = None, **kwargs):
        super().__init__(data)

        if type(args[0]) == torch.Tensor:
            self.__tensor = args[0]
        else:
            self.__tensor = torch.tensor(*args, **kwargs)

    def __getitem__(self, item):
        if self.data is not None:
            slice_data = {}
            for key in self.data.keys():
                if self.data[key].shape[0] == self.__tensor.shape[0]:
                    slice_data[key] = self.data[key][item]
                else:
                    slice_data[key] = self.data[key]
        else:
            slice_data = None

        return Tensor(self.__tensor[item], data=slice_data)

    def __getattr__(self, item):
        return getattr(self.__tensor, item)

    def __mul__(self, other):
        if isinstance(other, Wrapper):
            return self * other.item()
        elif isinstance(other, np.ndarray):
            return Tensor(self.__tensor * torch.tensor(other).to(self.__tensor))
        else:
            return Tensor(self.item() * other)

    def __add__(self, other):
        if isinstance(other, Wrapper):
            return self + other.item()
        elif isinstance(other, np.ndarray):
            return Tensor(self.__tensor + torch.tensor(other).to(self.__tensor))
        else:
            return Tensor(self.item() + other)

    def __matmul__(self, other):
        if isinstance(other, Wrapper):
            return self @ other.item()
        elif isinstance(other, np.ndarray):
            return Tensor(self.__tensor @ torch.tensor(other).to(self.__tensor))
        else:
            return Tensor(self.item() @ other)

    def __sub__(self, other):
        if isinstance(other, Wrapper):
            return self - other.item()
        elif isinstance(other, np.ndarray):
            return Tensor(self.__tensor - torch.tensor(other).to(self.__tensor))
        else:
            return Tensor(self.item() - other)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __rmatmul__(self, other):
        if isinstance(other, Wrapper):
            return other.item() @ self
        elif isinstance(other, np.ndarray):
            return Tensor(torch.tensor(other).to(self.__tensor) @ self.__tensor)
        else:
            return Tensor(other @ self.item())

    def __rsub__(self, other):
        if isinstance(other, Wrapper):
            return other.item() - self
        elif isinstance(other, np.ndarray):
            return Tensor(torch.tensor(other).to(self.__tensor) - self.__tensor)
        else:
            return Tensor(other - self.item())

    def reshape(self, *args, **kwargs):
        return Tensor(self.__tensor.reshape(*args, **kwargs))

    def item(self):
        return self.__tensor
