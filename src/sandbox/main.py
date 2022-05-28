import sys
from pathlib import Path

from sandbox.action_selection_rules.epsilon_greedy import EpsilonGreedyActionSelection


path = Path(__file__)
sys.path.append(str(path.parents[1].absolute()))

import gym
from sandbox.enviroments.twenty_forty_eight.env import TwentyFortyEightEnv

from sandbox.algorithms.td_zero.td_zero import TDZero
from sandbox.wrappers.stats_wrapper import StatsWrapper
from sandbox.algorithms.q_learning.qlearning import QLearning
from sandbox.action_selection_rules.ucb import UCB
from sandbox.wrappers.discrete_env_wrapper import DiscreteEnvironment
from sandbox.action_selection_rules.thompson_sampling import ThompsonSampling

import matplotlib.pyplot as plt


def main():

    # env = gym.make("CliffWalking-v0")
    env = TwentyFortyEightEnv()
    env = StatsWrapper(env)
    env = DiscreteEnvironment(env)
    
    policy = EpsilonGreedyActionSelection(0.1)
    algorithm = QLearning(0.5, 1, policy)
    agent = algorithm.run(10, env)
    env.to_gif()
    env.plot()
    env.close()


if __name__ == '__main__':
    main()
    