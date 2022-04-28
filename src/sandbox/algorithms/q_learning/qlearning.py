from email import policy
from typing import Generic

import numpy as np
from gym.core import ActType, ObsType
from sandbox.agents.action_value_agent import ActionValueAgent
from sandbox.algorithms.algorithm import Algorithm
from sandbox.policies.generic_policies import (ActionCandidate, GreedyPolicy,
                                               Policy)
from sandbox.wrappers.discrete_env_wrapper import DiscreteEnvironment


class QLearning(Algorithm[ObsType, ActType, ActionValueAgent]):

    def __init__(self, alpha: float, gamma: float, policy: Policy[ActType]) -> None:
        self._alpha = alpha
        self._gamma = gamma
        self._policy = policy

    def run(self, n_episodes: int, env: DiscreteEnvironment[ObsType, ActType]):
        action_value_estimates = np.random.random((env.n_observations, env.n_actions))
        for _ in range(n_episodes):
            from_observation, _ = env.reset(seed=42, return_info=True)
            is_done = False
            while not is_done:
                print(env.render('ansi'))
                best_action = self._policy(self._create_action_candidates(env, from_observation, action_value_estimates))
                next_observation, reward, is_done, info = env.step(best_action)
                expected_reward = self._calculate_expected_reward(env, from_observation, best_action, next_observation, reward, action_value_estimates)
                action_value_estimates[from_observation, best_action] = expected_reward
                from_observation = next_observation
            print('---------------------------------')
        return ActionValueAgent(
            action_value_estimates,
            self._policy
        )
                

    def _create_action_candidates(self, env: DiscreteEnvironment[ObsType, ActType], from_observation: ObsType, action_value_estimates) -> list[ActionCandidate[ActType]]:
        return [ActionCandidate(
                    a, action_value_estimates[from_observation, a]
                ) for a in env.actions()]

    def _calculate_expected_reward(self, env: DiscreteEnvironment[ObsType, ActType], from_observation: ObsType, action: ActType, 
                next_observation: ObsType, reward: float, action_value_estimates) -> float:
        off_policy = GreedyPolicy()
        best_next_action = off_policy(self._create_action_candidates(env, next_observation, action_value_estimates))
        return action_value_estimates[from_observation, action] + self._alpha * (reward + self._gamma * action_value_estimates[next_observation, best_next_action] - action_value_estimates[from_observation, action])
