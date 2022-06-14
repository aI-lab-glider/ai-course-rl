from __future__ import annotations
import logging
from typing import Callable, Generic, Tuple

import numpy as np
from gym.core import ActType, ObsType
from sandbox.action_selection_rules.greedy import GreedyActionSelection
from sandbox.policies.action_value_policy import ActionValuePolicy
from sandbox.algorithms.algorithm import Algorithm
from sandbox.action_selection_rules.generic import ActionSelectionRule
from sandbox.wrappers.discrete_env_wrapper import DiscreteEnvironment
from collections import defaultdict


class QLearning(Algorithm[ObsType, ActType, ActionValuePolicy]):

    def __init__(self, alpha: float, gamma: float, action_selection_rule: ActionSelectionRule[ActType]) -> None:
        self._alpha = alpha
        self._gamma = gamma
        self._action_selection_rule = action_selection_rule

    def run(self, n_episodes: int, env: DiscreteEnvironment[ObsType, ActType]):
        action_value_estimates = defaultdict(lambda: np.zeros(env.n_actions))
        for i in range(n_episodes):
            logging.info(f'Starting episode {i}')
            self._run_episode(env, action_value_estimates)
        return ActionValuePolicy(
            action_value_estimates,
            self._action_selection_rule
        )

    def _run_episode(self, env: DiscreteEnvironment[ObsType, ActType], action_value_estimates: dict[ActType, np.ndarray]):
        from_observation = env.reset(seed=42, return_info=False)
        is_done = False
        while not is_done:
            logging.debug(env.render('ansi'))
            best_action = self._action_selection_rule(
                action_value_estimates[from_observation])
            next_observation, reward, is_done, _ = env.step(best_action)
            expected_reward = self._calculate_expected_reward(
                env, from_observation, best_action, next_observation, reward, action_value_estimates)
            action_value_estimates[from_observation][best_action] = expected_reward
            from_observation = next_observation
        logging.debug('---------------------------------')

    def _calculate_expected_reward(self, env: DiscreteEnvironment[ObsType, ActType], previous_observation: ObsType, action: ActType,
                                   next_observation: ObsType, reward: float, action_value_estimates: dict[ObsType, np.ndarray[np._ShapeType, float]]) -> float:
        greedy_action_selection = GreedyActionSelection()
        best_next_action = greedy_action_selection(
            action_value_estimates[next_observation])
        td_target = reward + self._gamma * \
            action_value_estimates[next_observation][best_next_action]
        td_delta = td_target - \
            action_value_estimates[previous_observation][action]
        return action_value_estimates[previous_observation][action] + self._alpha * td_delta
