from hashlib import sha1
import logging

from numpy import exp2
from sandbox.action_selection_rules.greedy import GreedyActionSelection
from sandbox.algorithms.dqn.dqn import DQNAlgorithm
from sandbox.algorithms.dqn.policy import CartPoleQNetwork
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


class CartPoleWrapper(gym.Wrapper):
    """Wrapper for CartPole to make it compatible with tabular algorithms and avoid rendering problems"""

    def step(self, action):
        observation, reward, is_done, info = super().step(action)
        return tuple(observation), reward, is_done, info

    def reset(self, return_info=False, **kwargs):
        observation, info = super().reset(return_info=True, **kwargs)
        observation = tuple(observation)
        return (observation, info) if return_info else observation

    def render(self, mode="human", **kwargs):
        ...


if __name__ == '__main__':
    # NOTE: change logging level to info if you don't want to see ansi renders of env
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s\n%(message)s')
    cmp = Comparator()
    memory_size = 128
    envs_with_policices = cmp.compare_algorithms(algorithms=[
        QLearning(.9, .9, EpsilonGreedyActionSelection(.1)),
        DQNAlgorithm(
            memory_size,
            CartPoleQNetwork,
            EpsilonGreedyActionSelection(.1),
            learning_rate=0.1,
            max_episode_length=memory_size,
            exploration_steps=memory_size/2,
            target_update_steps=2*memory_size,
            batch_size=int(memory_size/8),
            discount_rate=.9
        )
    ],
        envs=[
            CartPoleWrapper(NamedEnv(f"Cart pole", DiscreteEnvironment(
                gym.make("CartPole-v1"))))],
        get_algorithm_label=lambda algo: type(algo).__name__,
        n_episodes=1000,
        plot_types=[PlotType.CumulatedReward, PlotType.RewardsVsEpNumber])
