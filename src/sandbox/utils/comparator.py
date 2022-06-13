from copy import deepcopy
from random import random
from typing import Callable
import gym
from matplotlib import pyplot as plt
from sandbox.algorithms.algorithm import Algorithm
from sandbox.wrappers.stats_wrapper import PlotType, StatsWrapper


class Comparator:
    def __init__(self, algorithms: list[Algorithm], envs: list[gym.Env], get_label: Callable[[Algorithm], str]) -> None:
        self.algorithms = algorithms
        self.envs = envs
        self.get_label = get_label

    def run(self, plot_types: list[PlotType]):
        _, axs = plt.subplots(len(self.envs), len(plot_types), figsize=(10, 10), squeeze=False)
        algo_colors = [(random(), random(), random()) for _ in self.algorithms]
        for i, env in enumerate(self.envs):
            env_axs = axs[i]
            for algo, color in zip(self.algorithms, algo_colors):
                env = deepcopy(env)
                env = StatsWrapper(env, False)
                _ = algo.run(100, env)
                env.plot(types=plot_types, ax=env_axs, color=color)
            for ax in env_axs:
                ax.legend([self.get_label(a) for a in self.algorithms])
        plt.ioff()
        plt.show()
