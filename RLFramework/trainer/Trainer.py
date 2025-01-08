import torch
import copy
from ..optim import Optimizer
from ..traj import Trajectory, ReplayMemory, Sample
from ..deeprl import *
from ..net import *
from ..utils import Logger


class Trainer(object):
    def __init__(self, agent: Agent, env: Environment, optimizers: list[Optimizer],
                 memory: ReplayMemory = None, logger: Logger = None, test_mode=False, **networks):
        self.agent = agent
        self.env = env

        self.optimizers = optimizers
        self.networks = networks

        self.losses = []
        for _ in self.optimizers:
            self.losses.append(0)

        self.timestep = 1
        self.episode = 1

        self.traj = Trajectory()
        self.memory = memory
        self.logger = logger

        self.test_mode = test_mode

        self.interval_functions = []
        self.interval_timers = []

        self.__perturbation = None

        self.__data = self.agent.policy_net.get_data()

        # add target networks
        for net_name in list(networks.keys()):
            _net = networks[net_name]
            if isinstance(_net, PolicyNet):
                _net.init_space(self.env.observation_space, self.env.action_space)
            elif isinstance(_net, ValueNet):
                _net.init_space(self.env.observation_space)
            elif isinstance(_net, QNet):
                _net.init_space(self.env.observation_space, self.env.action_space)
            else:
                raise TypeError("network must be one of Policy, Value, or Q net.")

            if _net.use_target:
                self.networks[net_name + "_target"] = _net.get_target_network()

        for optim in self.optimizers:
            optim.feed(self.networks)

    def add_interval(self, function, step=None, episode=None, min_step=0, min_episode=0,
                     args: list = None, kwargs: dict = None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        self.interval_functions.append((function, args, kwargs, min_step, min_episode, step, episode))
        self.interval_timers.append((0, 0))

    def __execute_interval(self, terminate):
        for i, (func, args, kwargs, min_step, min_episode, step, episode) in enumerate(self.interval_functions):
            if self.timestep < min_step or self.episode < min_episode:
                continue

            last_step, last_episode = self.interval_timers[i]

            if episode is None or terminate:
                if ((step is None or self.timestep - last_step >= step) and
                        (episode is None or self.episode - last_episode >= episode)):
                    func(*args, **kwargs)
                    self.interval_timers[i] = (self.timestep, self.episode)

    def step(self, perturbation=None):
        old_state = self.env.get_state()
        self.agent.set_state(old_state)

        self.agent.policy_net.set_data(**self.__data)
        action, logprob = self.agent.act()

        if perturbation is not None:
            self.env.act(action + perturbation)
        else:
            self.env.act(action)

        self.env.step()

        terminate = self.is_episode_done()

        self.traj.append(
            state=old_state,
            action=action,
            logprob=logprob,
            reward=self.env.get_reward(),
            termination=terminate,
            data=copy.deepcopy(self.__data)
        )

        self.__data = self.agent.policy_net.get_data()

        if self.memory is not None:
            self.memory.append_element(self.traj.recent())

        self.__execute_interval(terminate)

        if self.logger is not None:
            if ((self.logger.step_mode == "episode" and terminate)
                    or self.logger.step_mode == "step"):
                self.logger.step(self)

        if terminate:
            self.__reset()
            self.episode += 1
            self.__data = self.agent.policy_net.get_data()

        self.timestep += 1

    def step_optim(self, x):
        losses = []
        for optim in self.optimizers:
            losses.append(optim.step(x))

        self.losses = losses

    def train(self):
        if self.test_mode:
            return

        if self.memory is not None:
            x = self.memory.sample()
        else:
            x = Sample(self.traj.get_elements())

        self.step_optim(x)

    def __reset(self):
        self.env.reset()
        self.agent.reset()
        self.traj.reset()
        self.reset()

    def reset(self):
        pass

    def is_episode_done(self):
        return self.env.done

    def save(self, base_path: str = "./", version=0):
        for network in self.networks.keys():
            if "_target" not in network:
                torch.save(self.networks[network].state_dict(), base_path + "_" + network + f"_{version}.pth")

        if self.logger is not None:
            self.logger.save(base_path + f"_{version}_log.json")

    def load(self, base_path: str = "./", version=0):
        for network in self.networks.keys():
            if "_target" not in network:
                self.networks[network].load_state_dict(torch.load(base_path + "_" + network + f"_{version}.pth"))
            else:
                self.networks[network].load_state_dict(torch.load(base_path + "_" + network[:-7] + f"_{version}.pth"))

        if self.logger is not None:
            self.logger.load(base_path + f"_{version}_log.json")

    def add_perturbation(self, action):
        self.__perturbation = action

    def run(self, test_mode=False, max_step=None, max_episode=None):
        if self.logger is not None:
            self.logger.start_realtime_plot()

        temp_greedy = self.agent.greedy

        if test_mode:
            self.agent.greedy = True
            self.test_mode = True

        try:
            while True:
                perturbated = self.__perturbation is not None

                self.step(self.__perturbation)

                if perturbated:
                    self.__perturbation = None

                if max_step is not None and self.timestep > max_step:
                    print("max step reached")
                    break
                elif max_episode is not None and self.episode > max_episode:
                    print("max episode reached")
                    break

        except KeyboardInterrupt:
            print("KeyboardInterrupt")

        if self.logger is not None:
            self.logger.end_realtime_plot()

        if test_mode:
            self.agent.greedy = temp_greedy
            self.test_mode = False
