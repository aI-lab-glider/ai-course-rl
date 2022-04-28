from typing import Callable
from sandbox.algorithms.algorithm import Algorithm
import gym
import numpy as np
from sandbox.enviroments.base.env import Environment

from sandbox.enviroments.multi_armed_bandit.policy import DeterministicPolicy, Policy
from sandbox.value_estimates.tabular_value_estimates import TabularStateValueEstimates


class TDZero(Algorithm[TabularStateValueEstimates]):

    def __init__(self, alpha: float, gamma: float, sample_actions: int = 40) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.sample_actions = sample_actions

    def run(self, n_episodes: int, env: Environment, policy: DeterministicPolicy) -> TabularStateValueEstimates:
        state_value = TabularStateValueEstimates(env.all_observations(), lambda o: 0 if env.is_terminal_observation(o) else np.random.random())
        for _ in range(n_episodes):
            observation = env.reset(seed=42, return_info=False)
            done = False
            while not done:
                actions = np.unique(env.action_space_for_observation(observation).sample() for i in _ range(self.sample_actions))
                action = policy(observation, actions, state_value)
                new_observation, reward, done = env.step(action)
                state_value[observation] = \
                    state_value[observation] + self.alpha * (reward + self.gamma * state_value[new_observation] - state_value[observation])
                observation = new_observation
        return state_value