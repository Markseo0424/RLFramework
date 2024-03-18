import torch
import numpy as np
from .exploration import Exploration
from ..space import Space


class StochasticDiscrete(Exploration):
    def __init__(self, action_space: Space = None):
        super().__init__(action_space=action_space)

    def explore(self, x, numpy=False):
        assert x.shape[-len(self.policy_shape):] == self.policy_shape, f"x must have shape of {self.policy_shape}."
        assert type(x) == torch.Tensor, "input must be torch.Tensor."
        assert self.action_space is not None, "must init action space first!"

        if len(x.shape) == len(self.policy_shape):
            unbatched = True
            x = x.reshape((1, *self.policy_shape))
        else:
            unbatched = False

        x = torch.nn.functional.softmax(x, dim=1)

        dist = torch.distributions.Categorical(x)

        res = dist.sample()
        logprobs = dist.log_prob(res)

        if unbatched:
            res = res.reshape(res.shape[1:])
            logprobs = logprobs.reshape(logprobs.shape[1:])

        if numpy:
            res = res.detach().cpu().numpy()
            logprobs = logprobs.detach().cpu().numpy()

        return res, logprobs

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
