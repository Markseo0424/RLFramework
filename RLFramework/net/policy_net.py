import numpy as np
import torch
import torch.nn as nn
from .network import Network
from ..space import Space
from ..exp import Exploration
from ..wrapper import Wrapper


class PolicyNet(Network):
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

        self.policy_shape = self.exploration.policy_shape

    def init_space(self, observation_space: Space, action_space: Space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.exploration.init_space(self.action_space)
        self.policy_shape = self.exploration.policy_shape

    def sample_action(self, x, greedy=False, numpy=False):
        if greedy:
            return self.exploration.greedy(x, numpy=numpy)
        else:
            return self.exploration.explore(x, numpy=numpy)

    def get_logprob(self, x, action, numpy=False):
        if isinstance(action, Wrapper):
            action = action.item()
        return self.exploration.get_logprob(x, action, numpy=numpy)

    def __call__(self, state, eval=False, numpy=False):
        assert self.observation_space is not None and self.action_space is not None, \
            "must init observation, action space first!"

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

        # match to policy shape
        res = res.reshape((-1, *self.policy_shape))

        # if unbatched, flatten batch
        if unbatched:
            res = res.reshape(res.shape[1:])

        # check if policy shape is correct
        assert res.shape[-len(self.policy_shape):] == self.policy_shape, \
            f"expected policy shape as {self.policy_shape}, got {res.shape}"

        # if numpy is True, make to numpy array
        if numpy:
            res = res.detach().cpu().numpy()

        return res

    def forward(self, state):
        raise NotImplementedError

#
# class test_net(PolicyNet):
#     def __init__(self):
#         super().__init__(
#             optimizer=PolicyOptimizer([]),
#             state_space=Continuous(upper=np.array([1,1,1,1,1]), lower=np.array([-1,-1,-1,-1,-1])),
#             action_space=Discrete(6),
#             exploration=EpsilonGreedy(action_space=Discrete(6), epsilon=0.5)
#         )
#
#         self.module = nn.Sequential(
#             nn.Linear(5,10),
#             nn.ReLU(),
#             nn.Linear(10,10),
#             nn.ReLU(),
#             nn.Linear(10,6)
#         )
#
#     def forward(self, state):
#         return self.module(state)
