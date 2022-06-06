import logging
import random
from argparse import Action
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, NamedTuple, Protocol

import numpy as np
from gym.core import ActType, ObsType
from sandbox.action_selection_rules.greedy import GreedyActionSelection
from sandbox.policies.state_value_policy import StateValuePolicy
from sandbox.algorithms.algorithm import Algorithm
from sandbox.action_selection_rules.generic import (ActionCandidate,
                                               ActionSelectionRule)
from sandbox.wrappers.discrete_env_wrapper import DiscreteEnvironment



class TDZero(Algorithm[ObsType, ActType, StateValuePolicy]):

    def __init__(self, alpha: float, gamma: float, action_selection_rule: ActionSelectionRule[ActType]) -> None:
        self._alpha = alpha
        self._gamma = gamma
        self._action_selection_rule = action_selection_rule

    def run(self, n_episodes: int, env: DiscreteEnvironment[ObsType, ActType]):
        state_value_estimates = {}
        for _ in range(n_episodes):
            from_observation, info = env.reset(seed=42, return_info=True)
            is_done = False
            while not is_done:
                logging.debug(env.render('ansi'))
                action = self._action_selection_rule(self._get_action_candidates(env, state_value_estimates))
                next_observation, reward, is_done, info = env.step(action)
                expected_reward = self._calculate_expected_reward(from_observation, next_observation, reward, state_value_estimates)
                state_value_estimates[from_observation] = expected_reward
                from_observation = next_observation
        return StateValuePolicy(
            state_value_estimates,
            self._create_observation_to_action_mapping(env, state_value_estimates)
        )

    def _get_action_candidates(self, env: DiscreteEnvironment[ObsType, ActType], state_value_estimates: dict[ObsType, float]) -> ActionCandidate[ActType]:
        """Returns actions that could be selected from the current state with the estimated rewards could be obtained by taking an action"""
        candidates = []
        for action in env.actions():
            action_env = deepcopy(env)
            next_observation, _, is_done, info = action_env.step(action)
            candidates.append(ActionCandidate(
                action,
                0 if is_done else state_value_estimates.setdefault(next_observation, random.random())
            ))
        return candidates
        

    def _calculate_expected_reward(self, from_observation: ObsType, next_observation: ObsType, reward: float, state_value_estimates: dict[ObsType, float]):
        curr_observation_reward = state_value_estimates.setdefault(from_observation, random.random())
        td_delta = state_value_estimates.setdefault(next_observation, random.random()) - curr_observation_reward
        return curr_observation_reward + self._alpha * (reward + td_delta)

    def _create_observation_to_action_mapping(self, env: DiscreteEnvironment[ObsType, ActType], state_value_estimates: dict[ObsType, float]) -> dict[ObsType, ActType]:
        result = {}
        greedy_policy = GreedyActionSelection[ActType]()
        is_done = False
        from_observation = env.reset()
        while not is_done:
            action_candidates = self._get_action_candidates(env, state_value_estimates)
            optimal_action = greedy_policy(action_candidates)
            result[from_observation] = optimal_action
            from_observation, _, is_done, _ = env.step(optimal_action)
        return result



    


