import os
import time

import numpy as np
import matplotlib.pyplot as plt
import threading
import warnings
import json


class Logger(object):
    def __init__(self, realtime_plot=False, window_size=100, width=1, step_mode="episode", **kwargs):
        self.kwargs = kwargs
        self.enable_realtime_plot = realtime_plot
        self.window_size = window_size
        self.width = width

        assert step_mode == "episode" or step_mode == "step", "step mode must be either 'episode' or 'step'."
        self.step_mode = step_mode

        self.timestep = 0

        self.plots = {}

        for title in self.kwargs.keys():
            self.plots[title] = {}
            for line in self.kwargs[title].keys():
                self.plots[title][line] = []

        self.__thread = None

    def __getattr(self, obj, name: str):
        if not name:
            return obj

        if "." in name:
            name_list = name.split(".")
            _obj = obj

            for _name in name_list:
                _obj = self.__getattr(_obj, _name)

            return _obj

        else:
            bracket = False
            index = ""
            _obj = None

            for i, s in enumerate(name):
                if s == "[":
                    bracket = True
                    _obj = self.__getattr(obj, name[:i])

                elif bracket:
                    if s == "]":
                        bracket = False

                        if index.isdigit():
                            index = int(index)

                        return self.__getattr(_obj[index], name[i + 1:])

                    else:
                        index += s

                elif s == ")" and name[i - 1] == "(":
                    _obj = self.__getattr(obj, name[:i - 1])

                    return self.__getattr(_obj(), name[i + 1:])

        return getattr(obj, name)

    def step(self, obj):
        self.timestep += 1
        for title in self.plots.keys():
            for line in self.plots[title].keys():
                self.plots[title][line].append(self.__getattr(obj, self.kwargs[title][line]))

    def __realtime_plot(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            plot_num = 0
            plot_titles = []

            for title in self.plots.keys():
                if title[0] != "_":
                    plot_num += 1
                    plot_titles.append(title)

            fig, axes = plt.subplots(int(np.ceil(plot_num / self.width)), min(self.width, plot_num))

            if type(axes) == np.ndarray:
                axes = axes.reshape(-1)
            else:
                axes = np.array([axes])

            lines = {}

            for i, title in enumerate(plot_titles):
                lines[title] = {}
                for line in self.plots[title].keys():
                    if line[0] == "_":
                        continue

                    l, = axes[i].plot(self.plots[title][line], label=line)
                    lines[title][line] = l
                axes[i].legend()
                axes[i].set_title(title)
                axes[i].set_xlim(0, self.window_size)

            last_timestep = self.timestep

            while True:
                if self.timestep != last_timestep:
                    x_range = max(self.timestep - self.window_size, 0), max(self.timestep, self.window_size)
                    plot_range = x_range[0], self.timestep

                    for i, title in enumerate(lines.keys()):
                        plot_min = 1e+7
                        plot_max = -1e+7

                        for line in lines[title].keys():
                            l = lines[title][line]

                            line_data = self.plots[title][line][plot_range[0]: plot_range[1]]

                            l.set_data(
                                np.arange(plot_range[0], plot_range[1]),
                                line_data
                            )

                            line_min, line_max = min(line_data), max(line_data)
                            if plot_min > line_min:
                                plot_min = line_min
                            if plot_max < line_max:
                                plot_max = line_max

                        axes[i].set_xlim(x_range[0], x_range[1])
                        axes[i].set_ylim(plot_min, plot_max)

                plt.pause(0.1)

            plt.show()

    def start_realtime_plot(self):
        if not self.enable_realtime_plot:
            return

        self.__thread = threading.Thread(target=self.__realtime_plot)
        self.__thread.daemon = True
        self.__thread.start()

    def end_realtime_plot(self):
        if self.__thread is None:
            return

        self.__thread = None

    def save(self, path):
        jsonData = self.plots
        with open(path, 'w') as f:
            json.dump(jsonData, f, indent=2)

    def load(self, path):
        if not os.path.isfile(path):
            return

        with open(path, 'r') as f:
            self.plots = json.load(f)
            plot = self.plots[list(self.plots.keys())[0]]
            self.timestep = len(plot[list(plot.keys())[0]])
