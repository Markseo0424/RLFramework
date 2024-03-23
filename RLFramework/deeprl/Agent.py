from ..net import *


class Agent(object):
    def __init__(self, policy: PolicyNet | QNet, greedy=False):
        self._state = None
        self.policy_net = policy

        self.greedy = greedy

    def act(self):
        """
        Get an action about current state.
        It must be followed by set_state().
        :return: Current action about current state.
        """
        return self.policy(self._state)

    def reset(self):
        self._state = None
        self.policy_net.reset_data()
        self.reset_params()

    def set_state(self, state):
        """
        Setter of current state.
        :param state: Current state of Environment.
        """
        self._state = state

    def policy(self, state):
        """
        :param state: Current state of environment.
        :return: Action about current state.
        Returns action based on current state.
        Corresponds to pi(s,a).
        Action must be index of an action when using Deep RL Trainers.
        """
        policy = self.policy_net(state, eval=True)
        action ,logprob = self.policy_net.sample_action(policy, numpy=True, greedy=self.greedy)

        return action, logprob

    def reset_params(self):
        pass
