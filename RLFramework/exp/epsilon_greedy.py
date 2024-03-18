import torch
import numpy as np
from .exploration import Exploration
from ..space import Space


class EpsilonGreedy(Exploration):
    def __init__(self, action_space: Space = None, epsilon: float = 0.5,
                 epsilon_decay: float = 1, min_epsilon: float = 0):
        assert 0 <= epsilon <= 1, "epsilon must be in [0,1]."
        assert 0 <= epsilon_decay <= 1, "epsilon decay must be in [0,1]."
        assert 0 <= min_epsilon <= epsilon, "min epsilon must be in [0,epsilon]."

        super().__init__(action_space=action_space)

        self.init_epsilon = epsilon
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def explore(self, x, numpy=False):
        assert x.shape[-len(self.policy_shape):] == self.policy_shape, f"x must have shape of {self.policy_shape}."
        assert type(x) == torch.Tensor, "input must be torch.Tensor."
        assert self.action_space is not None, "must init action space first!"

        if len(x.shape) == len(self.policy_shape):
            unbatched = True
            x = x.reshape((1, *self.policy_shape))
        else:
            unbatched = False

        if np.random.random() < self.epsilon:
            if numpy:
                res = np.array(self.action_space.sample_idx(x.shape[0]))
            else:
                res = torch.tensor(self.action_space.sample_idx(x.shape[0])).to(x.device)
        else:
            if numpy:
                res = self.action_space.greedy(x).detach().cpu().numpy()
            else:
                res = self.action_space.greedy(x)

        if unbatched:
            res = res.reshape(res.shape[1:])

        return res, 0

    def greedy(self, x, numpy=False):
        if numpy:
            return self.action_space.greedy(x).detach().cpu().numpy(), 0
        else:
            return self.action_space.greedy(x), 0

    def get_logprob(self, x, action, numpy=False):
        assert x.shape[-len(self.policy_shape):] == self.policy_shape, f"x must have shape of {self.policy_shape}."
        assert type(x) == torch.Tensor, "input must be torch.Tensor."
        assert self.action_space is not None, "must init action space first!"

        if len(x.shape) == len(self.policy_shape):
            unbatched = True
            x = x.reshape((1, *self.policy_shape))
        else:
            unbatched = False

        assert x.shape[0] == action.shape[0], "batch size doesn't match!"

        probs = torch.nn.functional.softmax(x, dim=1)
        logprobs = torch.log(probs[np.arange(x.shape[0]), action] + 1e-7).reshape(-1, 1)
        entropy = torch.sum(probs * torch.log(probs), dim=1, keepdim=True)

        if unbatched:
            logprobs = logprobs.reshape(logprobs.shape[1:])
            entropy = entropy.reshape(entropy.shape[1:])

        if numpy:
            logprobs = logprobs.detach().cpu().numpy()
            entropy = entropy.detach().cpu().numpy()

        return logprobs, entropy

    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def reset(self):
        self.epsilon = self.init_epsilon
