import numpy as np
import gymnasium as gym
from .. import Environment
from ..space import Discrete, Continuous
from gymnasium.wrappers import RecordVideo


class GymnasiumEnvironment(Environment):
    def __init__(self, env_name: str, seed=None, render_mode=None,
                 record=False, record_path="./", truncate=True, **kwargs):
        if record:
            self.env = RecordVideo(gym.make(env_name, render_mode="rgb_array", **kwargs), video_folder=record_path)
        else:
            self.env = gym.make(env_name, render_mode=render_mode, **kwargs)
        observation_space = self.convert_space(self.env.observation_space)
        action_space = self.convert_space(self.env.action_space)

        super().__init__(observation_space=observation_space, action_space=action_space)

        self.seed = seed
        self.gym_reward = 0

        self.truncate = truncate

        self.episode_reward = 0

        self.__reset_params()

    def convert_space(self, gym_space):
        if isinstance(gym_space, gym.spaces.Discrete):
            return Discrete(gym_space.n)
        elif isinstance(gym_space, gym.spaces.Box):
            return Continuous(
                upper=np.array(gym_space.high),
                lower=np.array(gym_space.low)
            )
        else:
            raise NotImplementedError

    def update(self, state, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.gym_reward = reward
        end = terminated or (self.truncate and truncated)

        return observation, end

    def reward(self, state, action, next_state):
        # print(self.gym_reward)
        self.episode_reward += self.gym_reward

        return self.gym_reward

    def reset(self):
        super().reset()
        self.__reset_params()

    def __reset_params(self):
        self.gym_reward = 0
        self.episode_reward = 0

        observation, info = self.env.reset(seed=self.seed)
        self.init_state(observation)
