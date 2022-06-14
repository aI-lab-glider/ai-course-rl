import logging
import random
from sandbox.action_selection_rules.epsilon_greedy_with_epsilon_decay import EpsilonGreedyActionSelectionWithDecayEpsilon
from sandbox.action_selection_rules.greedy import GreedyActionSelection
from sandbox.algorithms.algorithm import Algorithm
from sandbox.algorithms.q_learning.qlearning import QLearning
from sandbox.algorithms.q_learning.dynaq import DynaQ
from sandbox.algorithms.q_learning.dynaq_plus import DynaQPlus
from sandbox.algorithms.q_learning.double_q import DoubleQLearning
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
        case DoubleQLearning():
            return f"DoubleQLearning(alpha={algorithm._alpha}, {type(algorithm._action_selection_rule).__name__})" 
        case QLearning():
            return f"QLearning(alpha={algorithm._alpha}, {type(algorithm._action_selection_rule).__name__})"        
        case DynaQ():
            return f"DynaQ(alpha={algorithm._alpha}, n={algorithm._planning_steps}, {type(algorithm._action_selection_rule).__name__})"
        case DynaQPlus():
            return f"DynaQPlus(alpha={algorithm._alpha}, n={algorithm._planning_steps}, kappa={algorithm._kappa}, {type(algorithm._action_selection_rule).__name__})"


def grid_pathinding_benchmark(n_episodes: int):
    cmp = Comparator()
    policies = cmp.compare_algorithms(
        algorithms=[
            QLearning(.1, 1, EpsilonGreedyActionSelection(.1)),
            QLearning(.1, 1, EpsilonGreedyActionSelectionWithDecayEpsilon(0.99, 0.99)),
            DynaQ(.1, 1, 50, EpsilonGreedyActionSelection(.1)),
            DynaQPlus(.1, 1, 50, 0.001, EpsilonGreedyActionSelection(.1)),
            DoubleQLearning(0.1, 1, EpsilonGreedyActionSelection(.1))
        ],
        envs=[
            NamedEnv(f"Grid pathfinding: {problem_size}", DiscreteEnvironment(
                gym.make("custom/gridpathfinding-v0",
                         file=f"{Path(gp.__file__).parent}/benchmarks/{problem_size}.txt")
            )) for problem_size in ["4", "16", "25"]],
        get_algorithm_label=_plot_name,
        n_episodes=n_episodes,
        plot_types=[PlotType.CumulatedReward])
    cmp.compare_policies(policies, 100)


def dynamic_pathinding_benchmark(n_episodes: int):
    cmp = Comparator()
    policies = cmp.compare_algorithms(
        algorithms=[
            DynaQPlus(.1, 1, 50, 0.01,
            EpsilonGreedyActionSelectionWithDecayEpsilon(0.99, 0.99)),
            DynaQ(.1, 1, 50, EpsilonGreedyActionSelectionWithDecayEpsilon(0.99, 0.99)),
            QLearning(.1, 1, EpsilonGreedyActionSelectionWithDecayEpsilon(0.99, 0.99)),
            QLearning(.1, 1, EpsilonGreedyActionSelection(0.05)),
            DoubleQLearning(0.1, 1, EpsilonGreedyActionSelection(.1))
            ],
        envs=[
            DiscreteEnvironment(gym.make(
            "custom/gridpathfinding-v0",
            file=f"{Path(gp.__file__).parent}/benchmarks/13_dynamic.txt",
            open_after=int(n_episodes / 3)))
        ],
        get_algorithm_label=_plot_name, n_episodes=n_episodes,
        plot_types=[PlotType.CumulatedReward])
    cmp.compare_policies(policies, 100)

def cliff_benchmark(n_episodes: int):
    cmp = Comparator()
    policies = cmp.compare_algorithms(
        algorithms=[
            QLearning(.1, 1, EpsilonGreedyActionSelection(.1)),
            QLearning(.1, 1, EpsilonGreedyActionSelectionWithDecayEpsilon(0.99, 0.99)),
            DoubleQLearning(.1, 1, EpsilonGreedyActionSelection(.1)),
            DynaQ(.1, 1, 50, EpsilonGreedyActionSelection(.1)),
            DynaQPlus(.1, 1, 50, 0.001, EpsilonGreedyActionSelection(.1)),
            DynaQ(.1, 1, 50, EpsilonGreedyActionSelectionWithDecayEpsilon(0.99, 0.99)),
            QLearning(.1, 1, EpsilonGreedyActionSelectionWithDecayEpsilon(0.99, 0.99))
        ],
        envs=[
            NamedEnv(f"Cliff walking", DiscreteEnvironment(gym.make("CliffWalking-v0")))
        ],
        get_algorithm_label=_plot_name,
        n_episodes=n_episodes,
        plot_types=[PlotType.CumulatedReward])
    cmp.compare_policies(policies, 100)

if __name__ == '__main__':
    # NOTE: change logging level to info if you don't want to see ansi renders of env
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s\n%(message)s')
    # grid_pathinding_benchmark(1000)
    # dynamic_pathinding_benchmark(2000)
    cliff_benchmark(1000)
