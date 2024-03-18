import numpy as np
import torch
import torch.nn as nn
import copy


class Network(nn.Module):
    def __init__(self, use_target=False, tau=1, **kwargs):
        super().__init__()

        self.use_target = use_target
        self.tau = tau

        self.target_network = None
        self.device = None

        self.set_data(init=True, **kwargs)

    def set_data(self, init=False, **kwargs):
        for var_name in kwargs.keys():
            if (not init) and (var_name not in self.__dict__.keys()):
                continue

            if isinstance(kwargs[var_name], np.ndarray):
                if init:
                    setattr(self, "__initial_" + var_name, np.copy(kwargs[var_name]))
                    setattr(self, var_name, kwargs[var_name])
                else:
                    init_var = getattr(self, "__initial_" + var_name)

                    if init_var is None:
                        setattr(self, var_name, kwargs[var_name])

                    if isinstance(init_var, np.ndarray):
                        setattr(self, var_name, kwargs[var_name])
                    elif isinstance(init_var, torch.Tensor):
                        setattr(self, var_name, torch.tensor(kwargs[var_name]).to(init_var))

            elif isinstance(kwargs[var_name], torch.Tensor):
                if init:
                    setattr(self, "__initial_" + var_name, torch.clone(kwargs[var_name]))
                    setattr(self, var_name, kwargs[var_name])
                else:
                    init_var = getattr(self, "__initial_" + var_name)

                    if init_var is None:
                        setattr(self, var_name, kwargs[var_name])

                    if isinstance(init_var, np.ndarray):
                        setattr(self, var_name, kwargs[var_name].detach().cpu().numpy())
                    elif isinstance(init_var, torch.Tensor):
                        setattr(self, var_name, kwargs[var_name].to(init_var))

            else:
                raise TypeError("type of network data should be numpy.ndarray or torch.Tensor.")

    def get_data(self):
        kwargs = self.__dict__
        data = {}

        for var_name in kwargs.keys():
            if var_name[:10] == "__initial_":
                init_var = kwargs[var_name]
                var = kwargs[var_name[10:]]

                if isinstance(var, np.ndarray):
                    if isinstance(init_var, torch.Tensor):
                        var = torch.tensor(var).to(init_var)
                elif isinstance(var, torch.Tensor):
                    if isinstance(init_var, torch.Tensor):
                        var = var.to(init_var)
                    else:
                        var = var.detach().cpu().numpy()

                else:
                    raise TypeError("type of network data should be numpy.ndarray or torch.Tensor.")

                data[var_name[10:]] = var

        return data

    def reset_data(self):
        kwargs = self.__dict__

        for var_name in kwargs.keys():
            if var_name[:10] == "__initial_":
                init_var = kwargs[var_name]
                if isinstance(init_var, torch.Tensor):
                    setattr(self, var_name[10:], torch.clone(init_var))
                else:
                    setattr(self, var_name[10:], np.copy(init_var))

    def get_target_network(self, tau=None):
        if tau is not None:
            self.tau = tau

        if self.target_network is not None:
            print("warning! target network already exist.")
        else:
            self.target_network = [copy.deepcopy(self)]

        return self.target_network[0]

    def update_target_network(self):
        assert self.target_network is not None, "there is no target network."
        # print("target update!")

        target_state_dict = self.target_network[0].state_dict()
        source_state_dict = self.state_dict()

        for param_name in source_state_dict:
            target_param = target_state_dict[param_name]
            source_param = source_state_dict[param_name]

            target_param.copy_(
                target_param * (1.0 - self.tau) + source_param * self.tau
            )

    def to(self, *args, device: torch.device, **kwargs):
        self.device = device
        return super().to(*args, device, **kwargs)
