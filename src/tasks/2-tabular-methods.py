import logging
import random
from sandbox.action_selection_rules.epsilon_greedy_with_epsilon_decay import EpsilonGreedyActionSelectionWithDecayEpsilon
from sandbox.action_selection_rules.greedy import GreedyActionSelection
from sandbox.algorithms.algorithm import Algorithm
from sandbox.algorithms.dyna.dynaq import DynaQ
from sandbox.algorithms.dyna.dynaq_plus import DynaQPlus
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


def _plot_name(algorithm: Algorithm) -> str:
    match algorithm:
        case QLearning():
            return f"QLearning(alpha={algorithm._alpha})"
        case DynaQ():
            return f"DynaQ(alpha={algorithm._alpha}, n={algorithm._planning_steps})"
        case DynaQPlus():
            return f"DynaQPlus(alpha={algorithm._alpha}, n={algorithm._planning_steps}, kappa={algorithm._kappa})"


def grid_pathinding_benchmark():
    cmp = Comparator()
    cmp = cmp.compare_algorithms(
        algorithms=[
            QLearning(.1, 1, EpsilonGreedyActionSelection(.1)),
            DynaQ(.1, 1, 50, EpsilonGreedyActionSelection(.1)),
            DynaQPlus(.1, 1, 50, 0.001, EpsilonGreedyActionSelection(.1))
        ],
        envs=[
            NamedEnv(f"Grid pathfinding: {problem_size}", DiscreteEnvironment(
                gym.make("custom/gridpathfinding-v0",
                         file=f"{Path(gp.__file__).parent}/benchmarks/{problem_size}.txt")
            )) for problem_size in ["4", "16", "25"]],
        get_algorithm_label=_plot_name,
        n_episodes=1000,
        plot_types=[PlotType.CumulatedReward])


def dynamic_pathinding_benchmark():
    n_episodes = 2000
    cmp = Comparator()
    cmp.compare_algorithms(
        algorithms=[
            DynaQPlus(.1, 1, 50, 0.01,
                      EpsilonGreedyActionSelectionWithDecayEpsilon(0.99, 0.99)),
            DynaQ(.1, 1, 50, EpsilonGreedyActionSelectionWithDecayEpsilon(0.99, 0.99)),
            QLearning(.1, 1, EpsilonGreedyActionSelectionWithDecayEpsilon(
                0.99, 0.99)),
        ],
        envs=[
            NamedEnv("Dynamic gridpathfinding: 13", DiscreteEnvironment(gym.make(
                "custom/gridpathfinding-v0",
                file=f"{Path(gp.__file__).parent}/benchmarks/13_dynamic.txt",
                open_after=int(n_episodes / 3))))
        ],
        get_algorithm_label=_plot_name, n_episodes=n_episodes,
        plot_types=[PlotType.CumulatedReward])


if __name__ == '__main__':
    # NOTE: change logging level to info if you don't want to see ansi renders of env
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s\n%(message)s')
    grid_pathinding_benchmark()
    dynamic_pathinding_benchmark()
