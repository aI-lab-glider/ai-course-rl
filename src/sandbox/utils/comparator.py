from copy import deepcopy
from random import random
from typing import Callable
import gym
from matplotlib import pyplot as plt
from sandbox.algorithms.algorithm import Algorithm
from sandbox.wrappers.stats_wrapper import PlotType, StatsWrapper
import distinctipy

class Comparator:
    def __init__(self, algorithms: list[Algorithm], envs: list[gym.Env], get_label: Callable[[Algorithm], str], n_episodes: int = 5000) -> None:
        self.algorithms = algorithms
        self.envs = envs
        self.get_label = get_label
        self.n_episodes = n_episodes

    def run(self, plot_types: list[PlotType]):
        _, axs = plt.subplots(len(self.envs), len(plot_types), figsize=(10, 10), squeeze=False)
        algo_colors = distinctipy.get_colors(len(self.algorithms))
        for i, env in enumerate(self.envs):
            env_axs = axs[i]
            for algo, color in zip(self.algorithms, algo_colors):
                env_copy = deepcopy(env)
                env_copy = StatsWrapper(env_copy)
                _ = algo.run(self.n_episodes, env_copy)
                env_copy.plot(types=plot_types, ax=env_axs, color=color)
            for ax in env_axs:
                ax.legend([self.get_label(a) for a in self.algorithms])
        plt.show()
