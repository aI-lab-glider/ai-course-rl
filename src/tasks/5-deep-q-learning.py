import logging
import random
from sandbox.action_selection_rules.greedy import GreedyActionSelection
from sandbox.algorithms.dqn.dqn import DQNAlgorithm
from sandbox.algorithms.dqn.policy import MyQNetwork
from sandbox.algorithms.q_learning.qlearning import QLearning
from sandbox.algorithms.td_zero.td_zero import TDZero
from sandbox.enviroments.multi_armed_bandit.env import NormalDistribution
from sandbox.policies.nn_policy import QNetwork
from sandbox.utils.comparator import Comparator
from sandbox.wrappers.discrete_env_wrapper import DiscreteEnvironment
from sandbox.wrappers.stats_wrapper import PlotType
import sandbox.enviroments
import gym
from sandbox.action_selection_rules.epsilon_greedy import EpsilonGreedyActionSelection
from sandbox.action_selection_rules.ucb import UCB

from sandbox.algorithms.bandits_algorithm.bandits_algorithm import BanditsAlgorithm
import sandbox.enviroments.grid_pathfinding as gp
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential


def cartpole_benchmark():
    logging.basicConfig(level=logging.INFO)

    cmp = Comparator(
        algorithms=[
            DQNAlgorithm(
                memory_size=2000,
                create_network=MyQNetwork,
                action_selection_rule=EpsilonGreedyActionSelection(0.1),
                learning_rate=1e-2,
                max_episode_length=200,
                exploration_steps=50,
                target_update_steps=50,
                batch_size=64,
                discount_rate=0.99,
            )
            ],
        envs=[
        #    gym.make("CartPole-v1"),
           gym.make('LunarLander-v2')
        ],
        get_label=lambda algo: f"{type(algo).__name__}", n_episodes=1000
    )
    cmp.run(list(PlotType))

if __name__ == '__main__':
    # NOTE: change logging level to info if you don't want to see ansi renders of env
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s\n%(message)s')
    cartpole_benchmark()

