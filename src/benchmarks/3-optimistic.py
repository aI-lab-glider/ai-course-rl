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
from tasks._tabular_benchmarks import _plot_name
from sandbox.algorithms.bandits_algorithm.bandits_algorithm import BanditsAlgorithm
import sandbox.enviroments.grid_pathfinding as gp
from pathlib import Path

def cliff_benchmark(n_episodes: int):
    cmp = Comparator()
    policies = cmp.compare_algorithms(
        algorithms=[
            QLearning(.1, 1, EpsilonGreedyActionSelection(.1)),
            DoubleQLearning(.1, 1, EpsilonGreedyActionSelection(.1))
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
    cliff_benchmark(1000)
