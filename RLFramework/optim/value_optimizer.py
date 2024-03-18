import torch
from .optimizer import Optimizer


class ValueOptimizer(Optimizer):
    def __init__(self, required_list: list):
        super().__init__(required_list=required_list)
