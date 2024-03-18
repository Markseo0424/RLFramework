import numpy as np
import torch
import torch.nn as nn
from .network import Network
from ..space import *
from ..exp import Exploration
from ..wrapper import Wrapper


class QNet(Network):
    def __init__(self, *args, observation_space: Space = None, action_space: Space = None,
                 exploration: Exploration = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = observation_space
        self.action_space = action_space

        if exploration is None:
            self.exploration = Exploration(self.action_space)
        else:
            self.exploration = exploration
            if self.action_space is not None:
                self.exploration.init_space(self.action_space)

    def init_space(self, observation_space: Space, action_space: Space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.exploration.init_space(self.action_space)

    def sample_action(self, x, greedy=False, numpy=False):
        if greedy:
            return self.exploration.greedy(x, numpy=numpy)
        else:
            return self.exploration.explore(x, numpy=numpy)

    def __call__(self, state, action=None, eval=False, numpy=False):
        assert self.observation_space is not None and self.action_space is not None, \
            "must init observation, action space first!"

        # check if continuous and action is none.
        assert type(self.action_space) == Discrete or action is not None, \
            "action can't be None for Continuous action space."

        # if wrapped, unwrap
        if isinstance(state, Wrapper):
            self.set_data(**state.data)
            state = state.item()
        if isinstance(action, Wrapper):
            action = action.item()

        # convert state to FloatTensor
        if type(state) == np.ndarray:
            state = torch.FloatTensor(state).to(self.device)
        elif type(state) == torch.Tensor:
            state = state.to(self.device)
        else:
            assert False, "state must be numpy.ndarray or torch.Tensor."

        # convert action to FloatTensor (if action is Continuous)
        if type(self.action_space) == Continuous:
            if type(action) == np.ndarray:
                action = torch.FloatTensor(action).to(self.device)
            elif type(action) == torch.Tensor:
                action = action.to(self.device)
            else:
                assert False, "in Continuous action space, action must be numpy.ndarray or torch.Tensor."

        # check if state shape is correct
        assert state.shape[-len(self.observation_space.shape):] == self.observation_space.shape, \
            f"state dimension mismatch : expected {(-1, *self.observation_space.shape)}, but got {state.shape}."

        # check if continuous action shape is correct
        assert type(self.action_space) is not Continuous or action.shape[-len(self.action_space.shape):] == self.action_space.shape, \
            f"action dimension mismatch : expected {(-1, *self.action_space.shape)}, but got {action.shape}"

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
                res = super().__call__(state, action)
            self.train()
        else:
            res = super().__call__(state, action)

        # if discrete, index Q value
        if type(self.action_space) == Discrete and action is not None:
            assert res.shape[0] == action.shape[0], "action batch len mismatch."
            res = res[torch.arange(res.shape[0]), action.reshape((action.shape[0],))]

        # if unbatched, flatten batch
        if unbatched:
            res = res.reshape(res.shape[1:])

        # if numpy is True, make to numpy array
        if numpy:
            res = res.detach().cpu().numpy()

        return res

    def forward(self, state, action):
        raise NotImplementedError

#
# class test_net(QNet):
#     def __init__(self):
#         super().__init__(
#             optimizer=QOptimizer([]),
#             state_space=Continuous(
#                 upper=np.array([1,1,1,1,1]),
#                 lower=np.array([-1,-1,-1,-1,-1])
#             ),
#             action_space=Continuous(
#                 upper=np.array([1,1,1]),
#                 lower=np.array([-1,-1,-1])
#             )
#         )
#
#         self.model = nn.Sequential(
#             nn.Linear(8, 1)
#         )
#
#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#
#         return self.model(x)