import logging
from random import random
from sandbox.action_selection_rules.greedy import GreedyActionSelection
from sandbox.action_selection_rules.thompson_sampling import ThompsonSampling
from sandbox.enviroments.multi_armed_bandit.env import NormalDistribution
from sandbox.utils.comparator import Comparator
from sandbox.wrappers.named_env_wrapper import NamedEnv
from sandbox.wrappers.stats_wrapper import PlotType
import sandbox.enviroments
import gym
from sandbox.action_selection_rules.epsilon_greedy import EpsilonGreedyActionSelection
from sandbox.action_selection_rules.ucb import UCB

from sandbox.algorithms.bandits_algorithm.bandits_algorithm import BanditsAlgorithm


def _plot_name(algorithm: BanditsAlgorithm) -> str:
    match algorithm._select_action:
        case GreedyActionSelection():
            return "greedy"
        case EpsilonGreedyActionSelection(eps):
            return f"greedy(eps = {eps})"
        case UCB(c):
            return f"UCB(c = {c})"
        case ThompsonSampling():
            return f"Thompson"


if __name__ == '__main__':
    # NOTE: change logging level to info if you don't want to see ansi renders of env
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s\n%(message)s')
    cmp = Comparator()
    env_policies = cmp.compare_algorithms(
        algorithms=[
            BanditsAlgorithm(action_selection_rule)
            for action_selection_rule in
            [
                GreedyActionSelection(),
                EpsilonGreedyActionSelection(0.01),
                EpsilonGreedyActionSelection(0.1),
                EpsilonGreedyActionSelection(0.5),
                UCB(0.4),
                UCB(1.414),
                UCB(10),
                ThompsonSampling()]
        ],
        envs=[NamedEnv("100-armed bandits", gym.make(
            "custom/multiarmed-bandits-v0",
            reward_distributions=[NormalDistribution(
                random(), random()) for _ in range(100)]
        ))],
        get_algorithm_label=_plot_name,
        n_episodes=5000,
        plot_types=[PlotType.CumulatedReward])
    cmp.compare_policies(env_policies, n_episodes=100, max_episode_length=1)
