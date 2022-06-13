from copy import deepcopy
from itertools import product
import logging
from random import randint, random, seed
import sys
from pathlib import Path
from typing import Callable, Iterable
from matplotlib import pyplot as plt
import numpy as np
import gym


path = Path(__file__)
sys.path.append(str(path.parents[1].absolute()))
from sandbox.action_selection_rules.ucb import UCB

from sandbox.algorithms.algorithm import Algorithm
import sandbox.enviroments
from sandbox.action_selection_rules.epsilon_greedy import EpsilonGreedyActionSelection
from sandbox.algorithms.dqn import DQNAlgorithm, MyQNetwork, policy
from sandbox.algorithms.q_learning.qlearning import QLearning
from sandbox.wrappers.discrete_env_wrapper import DiscreteEnvironment
from sandbox.wrappers.stats_wrapper import PlotType, StatsWrapper
from sandbox.algorithms.bandits_algorithm.bandits_algorithm import BanditsAlgorithm
from sandbox.enviroments.multi_armed_bandit.env import NormalDistribution


class Comparer:
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
                env = StatsWrapper(env)
                _ = algo.run(5000, env)
                env.plot(types=plot_types, ax=env_axs, color=color)
            for ax in env_axs:
                ax.legend([self.get_label(a) for a in self.algorithms])
        plt.show()


def main():
    # NOTE: change logging level to info if you don't want to see ansi renders of env
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s\n%(message)s')
    cmp = Comparer(
        algorithms=[BanditsAlgorithm(UCB()), BanditsAlgorithm(EpsilonGreedyActionSelection(0.01))],
        envs=[gym.make(
            "custom/multiarmed-bandits-v0",
            reward_distributions=[NormalDistribution(random(), random()) for _ in range(5)]
            )
        ],
        get_label=lambda algo: type(algo._select_action).__name__
    )
    cmp.run([PlotType.CumRewardvsTime, PlotType.EpisodeLengthHist])

def enjoy(env, action_selection, steps) -> None:
    state = env.reset()
    for step in range(steps):
        action = action_selection(state)
        state, reward, done, info = env.step(action)
        if done:
            state = env.reset()
        img = env.render()



if __name__ == "__main__":
    main()
