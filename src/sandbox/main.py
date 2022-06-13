import logging
from random import random, seed
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import gym

path = Path(__file__)
sys.path.append(str(path.parents[1].absolute()))
from sandbox.action_selection_rules.ucb import UCB

import sandbox.enviroments
from sandbox.action_selection_rules.epsilon_greedy import EpsilonGreedyActionSelection
from sandbox.algorithms.dqn import DQNAlgorithm, MyQNetwork, policy
from sandbox.algorithms.q_learning.qlearning import QLearning
from sandbox.wrappers.discrete_env_wrapper import DiscreteEnvironment
from sandbox.wrappers.stats_wrapper import PlotType, StatsWrapper
from sandbox.algorithms.bandits_algorithm.bandits_algorithm import BanditsAlgorithm
from sandbox.enviroments.multi_armed_bandit.env import NormalDistribution


def main():
    # NOTE: change logging level to info if you don't want to see ansi renders of env
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s\n%(message)s')
    seed(42)
    env = gym.make(
        "custom/multiarmed-bandits-v0",
        reward_distributions=[NormalDistribution(random(), random()) for _ in range(5)]
    )
    env = StatsWrapper(env)
    algorithm = BanditsAlgorithm(UCB())
    policy = algorithm.run(5000, env)
    env.plot(types=[PlotType.CumRewardvsTime])
    plt.show()

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
