import logging
import random
from sandbox.action_selection_rules.greedy import GreedyActionSelection
from sandbox.algorithms.q_learning.qlearning import QLearning
from sandbox.algorithms.td_zero.td_zero import TDZero
from sandbox.enviroments.multi_armed_bandit.env import NormalDistribution
from sandbox.utils.comparator import Comparator
from sandbox.wrappers.discrete_env_wrapper import DiscreteEnvironment
from sandbox.wrappers.stats_wrapper import PlotType
import sandbox.enviroments
import gym
from sandbox.action_selection_rules.epsilon_greedy import EpsilonGreedyActionSelection
from sandbox.action_selection_rules.ucb import UCB

from sandbox.algorithms.bandits_algorithm.bandits_algorithm import BanditsAlgorithm
import sandbox.enviroments.grid_pathfinding as gp
from pathlib import Path

if __name__ == '__main__':
    # NOTE: change logging level to info if you don't want to see ansi renders of env
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s\n%(message)s')
    cmp = Comparator(
        algorithms=[
            QLearning(.01, 1, EpsilonGreedyActionSelection(.1)),
            QLearning(.05, 1, EpsilonGreedyActionSelection(.1)),
            QLearning(.1, 1, EpsilonGreedyActionSelection(.1)),
            ],
        envs=[
            DiscreteEnvironment(gym.make(
            "custom/gridpathfinding-v0",
            file=f"{Path(gp.__file__).parent}/benchmarks/4.txt")),
            DiscreteEnvironment(gym.make(
            "custom/gridpathfinding-v0",
            file=f"{Path(gp.__file__).parent}/benchmarks/16.txt")),
            DiscreteEnvironment(gym.make(
            "custom/gridpathfinding-v0",
            file=f"{Path(gp.__file__).parent}/benchmarks/25.txt")),
        ],
        get_label=lambda algo: f"{algo._alpha}", n_episodes=10000
    )
    cmp.run(list(PlotType))
