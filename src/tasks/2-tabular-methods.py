import logging
import random
from sandbox.action_selection_rules.greedy import GreedyActionSelection
from sandbox.algorithms.q_learning.qlearning import QLearning
from sandbox.algorithms.td_zero.td_zero import TDZero
from sandbox.enviroments.multi_armed_bandit.env import NormalDistribution
from sandbox.utils.comparator import Comparator
from sandbox.wrappers.discrete_env_wrapper import DiscreteEnvironment
from sandbox.wrappers.named_env_wrapper import NamedEnv
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
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s\n%(message)s')
    cmp = Comparator()
    envs_with_policices = cmp.compare_algorithms(algorithms=[
        QLearning(.01, 1, EpsilonGreedyActionSelection(.1)),
        QLearning(.05, 1, EpsilonGreedyActionSelection(.1)),
        QLearning(.1, 1, EpsilonGreedyActionSelection(.1)),
    ],
        envs=[
            NamedEnv(f"Grid pathfinding: {problem_size}", DiscreteEnvironment(
                gym.make("custom/gridpathfinding-v0",
                         file=f"{Path(gp.__file__).parent}/benchmarks/{problem_size}.txt")
            )) for problem_size in ["4", "16", "25"]],
        get_algorithm_label=lambda algo: f"{algo._alpha}",
        n_episodes=10_000,
        plot_types=list(PlotType))
    cmp.compare_policies(envs_with_policices, 100)
