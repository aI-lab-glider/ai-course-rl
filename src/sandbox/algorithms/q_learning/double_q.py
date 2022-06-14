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
import random

class DoubleQLearning(Algorithm[ObsType, ActType, ActionValuePolicy]):

    def __init__(self, alpha: float, gamma: float, action_selection_rule: ActionSelectionRule[ActType]) -> None:
        self._alpha = alpha
        self._gamma = gamma
        self._action_selection_rule = action_selection_rule

    def run(self, n_episodes: int, env: DiscreteEnvironment[ObsType, ActType]):
        action_value_estimates_a = defaultdict(lambda: np.zeros(env.n_actions))
        action_value_estimates_b = defaultdict(lambda: np.zeros(env.n_actions))
        for i in range(n_episodes):
            logging.info(f'Starting episode {i}')
            self._run_episode(env, action_value_estimates_a, action_value_estimates_b)
        return ActionValuePolicy(
            action_value_estimates_a,
            self._action_selection_rule
        )

    def _run_episode(self, env: DiscreteEnvironment[ObsType, ActType], action_value_estimates_a: dict[ActType, np.ndarray], action_value_estimates_b: dict[ActType, np.ndarray]):
        from_observation = env.reset(seed=42, return_info=False)
        is_done = False
        while not is_done:
            logging.debug(env.render('ansi'))
            # TODO:
            # 1. get action via self._action_selection_rule based on the average (sum?) of both estimates
            # 2. gather a new observation via env.step
            # 3. throw a coin:
            #    - if there is a tail: update expected_reward_a using expected_reward_b as a critic
            #    - if there is a head: update expected_reward_b using expected_reward_a as a critic
            # 4. update from_observation!
            best_action = self._action_selection_rule(
                (action_value_estimates_a[from_observation] + action_value_estimates_b[from_observation]))
            next_observation, reward, is_done, _ = env.step(best_action)

            if random.random() > 0.5:
                expected_reward_a = self._calculate_expected_reward(
                    from_observation, best_action, next_observation, reward, action_value_estimates_a, action_value_estimates_b)
                action_value_estimates_a[from_observation][best_action] = expected_reward_a
            else:
                expected_reward_b = self._calculate_expected_reward(
                    from_observation, best_action, next_observation, reward, action_value_estimates_b, action_value_estimates_a)
                action_value_estimates_b[from_observation][best_action] = expected_reward_b
            from_observation = next_observation
        logging.debug('---------------------------------')

    def _calculate_expected_reward(self, previous_observation: ObsType, action: ActType,
                                   next_observation: ObsType, reward: float, 
                                   learner: dict[ObsType, np.ndarray[np._ShapeType, float]],
                                   critic: dict[ObsType, np.ndarray[np._ShapeType, float]]) -> float:
        # TODO:
        # 1. choose action using greedy policy based on the learner estimated at the next state
        # 2. find td_target based on the critic(!) estimates
        # 3. calculate td_error compared to the learner estimates
        # 4. return learner estimates slightly (selg._alpha) nudged in the error direction
        greedy_action_selection = GreedyActionSelection()
        best_next_action = greedy_action_selection(
            learner[next_observation])
        td_target = reward + self._gamma * \
            critic[next_observation][best_next_action]
        td_error = td_target - \
            learner[previous_observation][action]
        return learner[previous_observation][action] + self._alpha * td_error
