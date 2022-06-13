import logging
from random import random
from sandbox.action_selection_rules.greedy import GreedyActionSelection
from sandbox.enviroments.multi_armed_bandit.env import NormalDistribution
from sandbox.utils.comparator import Comparator
from sandbox.wrappers.stats_wrapper import PlotType
import sandbox.enviroments
import gym
from sandbox.action_selection_rules.epsilon_greedy import EpsilonGreedyActionSelection
from sandbox.action_selection_rules.ucb import UCB

from sandbox.algorithms.bandits_algorithm.bandits_algorithm import BanditsAlgorithm


if __name__ == '__main__':
    # NOTE: change logging level to info if you don't want to see ansi renders of env
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s\n%(message)s')
    cmp = Comparator(
        algorithms=[
            BanditsAlgorithm(action_selection_rule)
            for action_selection_rule in 
                [GreedyActionSelection(),EpsilonGreedyActionSelection(0.01), EpsilonGreedyActionSelection(0.1), UCB()]
            ],
        envs=[gym.make(
            "custom/multiarmed-bandits-v0",
            reward_distributions=[NormalDistribution(random(), random()) for _ in range(5)]
            )
        ],
        get_label=lambda algo: type(algo._select_action).__name__
    )
    cmp.run([PlotType.CumulatedReward])
