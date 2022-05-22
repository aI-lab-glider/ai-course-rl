import sys
from pathlib import Path

path = Path(__file__)
sys.path.append(str(path.parents[1].absolute()))

import gym
from sandbox.algorithms.td_zero.td_zero import TDZero
from sandbox.wrappers.stats_wrapper import StatsWrapper
from sandbox.policies.generic_policies import EpsilonGreedyPolicy, GreedyPolicy
from sandbox.algorithms.q_learning.qlearning import QLearning

from sandbox.wrappers.discrete_env_wrapper import DiscreteEnvironment

import matplotlib.pyplot as plt


def main():

    env = gym.make("CliffWalking-v0")
    env = StatsWrapper(env)
    env = DiscreteEnvironment(env)
    
    policy = EpsilonGreedyPolicy(0.1)
    algorithm = QLearning(0.5, 1, policy)
    agent = algorithm.run(500, env)
    
    cumulated_reward = [s.cumulative_reward for s in env.stats]
    plt.plot(cumulated_reward)
    plt.show()
    env.close()


if __name__ == '__main__':
    main()
    