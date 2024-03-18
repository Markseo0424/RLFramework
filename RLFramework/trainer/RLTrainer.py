import torch
import copy
from ..optim import Optimizer
from ..traj import Trajectory, ReplayMemory, Sample
from ..deeprl import *
from ..net import *
from ..utils import Logger


class RLTrainer(object):
    def __init__(self, agent: RLAgent, env: RLEnvironment, optimizers: list[Optimizer],
                 memory: ReplayMemory = None, logger: Logger = None, **networks):
        self.agent = agent
        self.env = env

        self.optimizers = optimizers
        self.networks = networks

        self.timestep = 1
        self.episode = 1

        self.traj = Trajectory()
        self.memory = memory
        self.logger = logger

        self.interval_functions = {"step": [], "episode": []}

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

    def add_interval(self, function, step=None, episode=None, minimum=0):
        assert step is None and episode is not None or \
               step is not None and episode is None, "least one should be not None."

        if step is not None:
            self.interval_functions["step"].append((minimum, step, function))
        else:
            self.interval_functions["episode"].append((minimum, episode, function))

    def __execute_interval(self, terminate):
        for minimum, interval, func in self.interval_functions["step"]:
            if self.timestep >= minimum and self.timestep % interval == 0:
                func()

        if terminate:
            for minimum, interval, func in self.interval_functions["episode"]:
                if self.episode >= minimum and self.episode % interval == 0:
                    func()

    def step(self):
        old_state = self.env.get_state()
        self.agent.set_state(old_state)

        self.agent.policy_net.set_data(**self.__data)
        action, logprob = self.agent.act()

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
        for optim in self.optimizers:
            optim.step(x)

    def train(self):
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

    def save(self, base_path: str = "./", version: int = 0):
        for network in self.networks.keys():
            if "_target" not in network:
                torch.save(self.networks[network].state_dict(), base_path + network + f"_{version}.pth")

        if self.logger is not None:
            self.logger.save(base_path + f"{version}_log.json")

    def load(self, base_path: str = "./", version: int = 0):
        for network in self.networks.keys():
            if "_target" not in network:
                self.networks[network].load_state_dict(torch.load(base_path + network + f"_{version}.pth"))
            else:
                self.networks[network].load_state_dict(torch.load(base_path + network[:-7] + f"_{version}.pth"))

        if self.logger is not None:
            self.logger.load(base_path + f"{version}_log.json")

    def run(self, max_step=None, max_episode=None):
        if self.logger is not None:
            self.logger.start_realtime_plot()

        try:
            while True:
                self.step()
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
