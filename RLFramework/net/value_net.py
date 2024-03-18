import numpy as np
import torch
import torch.nn as nn
from .network import Network
from ..space import Space
from ..wrapper import Wrapper


class ValueNet(Network):
    def __init__(self, *args, observation_space: Space = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = observation_space

    def init_space(self, observation_space: Space):
        self.observation_space = observation_space

    def __call__(self, state, eval=False, numpy=False):
        assert self.observation_space is not None, \
            "must init observation space first!"

        # if wrapped, unwrap
        if isinstance(state, Wrapper):
            self.set_data(**state.data)
            state = state.item()

        # convert state to FloatTensor
        if type(state) == np.ndarray:
            state = torch.FloatTensor(state).to(self.device)
        elif type(state) == torch.Tensor:
            state = state.to(self.device)
        else:
            assert False, "state must be numpy.ndarray or torch.Tensor."

        # check if state shape is correct
        assert state.shape[-len(self.observation_space.shape):] == self.observation_space.shape, \
            f"dimension mismatch : expected {(-1, *self.observation_space.shape)}, but got {state.shape}."

        # if unbatched, unflatten batch
        if len(state.shape) == len(self.observation_space.shape):
            unbatched = True
            state = state.reshape((1, *self.observation_space.shape))
        else:
            unbatched = False

        # calculate
        if eval:
            self.eval()
            with torch.no_grad():
                res = super().__call__(state)
            self.train()

        else:
            res = super().__call__(state)

        # if unbatched, flatten batch
        if unbatched:
            res = res.reshape(res.shape[1:])

        # if numpy is True, make to numpy array
        if numpy:
            res = res.detach().cpu().numpy()

        return res

    def forward(self, state):
        raise NotImplementedError
