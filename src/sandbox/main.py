import logging
import sys
from pathlib import Path
import numpy as np
import gym
path = Path(__file__)
sys.path.append(str(path.parents[1].absolute()))

import sandbox.enviroments
from sandbox.action_selection_rules.epsilon_greedy import EpsilonGreedyActionSelection
from sandbox.algorithms.dqn import DQNAlgorithm, MyQNetwork, policy
from sandbox.algorithms.q_learning.qlearning import QLearning
from sandbox.wrappers.discrete_env_wrapper import DiscreteEnvironment
from sandbox.wrappers.stats_wrapper import StatsWrapper


def main():
    # NOTE: change logging level to info if you don't want to see ansi renders of env
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s\n%(message)s')
    env = gym.make("custom/gridpathfinding-v0",
        file="src/sandbox/enviroments/grid_pathfinding/bencmarks/22.txt"
    )
    env = DiscreteEnvironment(env)
    env = StatsWrapper(env)
    algorithm = QLearning(0.1, 0.1, EpsilonGreedyActionSelection(0.05))
    policy = algorithm.run(1000, env)
    env.plot()
    

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
