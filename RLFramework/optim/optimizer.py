import torch
from ..traj import Sample


class Optimizer(object):
    def __init__(self, required_list: list):
        super().__init__()

        self.required_list = required_list

    def feed(self, networks: dict):
        for network_name in self.required_list:
            assert network_name in networks.keys(), f"required '{network_name}' is not in networks."
            setattr(self, network_name, networks[network_name])

        self.init_optim()

    def init_optim(self):
        pass

    def step(self, x: Sample):
        pass
