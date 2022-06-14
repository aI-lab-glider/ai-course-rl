from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Generic, NamedTuple

import numpy as np
from gym.core import ActType, ObsType
from sandbox.action_selection_rules.greedy import GreedyActionSelection
from sandbox.policies.action_value_policy import ActionValuePolicy
from sandbox.algorithms.algorithm import Algorithm
from sandbox.action_selection_rules.generic import ActionSelectionRule
from sandbox.wrappers.discrete_env_wrapper import DiscreteEnvironment
from collections import defaultdict
import random

DynaEntry = NamedTuple("DynaEntry", [("next_observation", ObsType), ("reward", float), ("is_done", bool)])
DynaMemory = NamedTuple("DynaMemory", [("from_observation", ObsType), ("action", ActType), ("next_observation", ObsType), ("reward", float), ("is_done", bool)])
DynaModel = dict[ObsType, dict[ActType, DynaEntry]]


class DynaQ(Algorithm[ObsType, ActType, ActionValuePolicy]):

    def __init__(self, alpha: float, gamma: float, planning_steps: int, action_selection_rule: ActionSelectionRule[ActType]) -> None:
        self._alpha = alpha
        self._gamma = gamma
        self._policy = action_selection_rule
        self._planning_steps = planning_steps
        

    def run(self, n_episodes: int, env: DiscreteEnvironment[ObsType, ActType]):
        model: DynaModel = dict()
        action_value_estimates = defaultdict(lambda: np.zeros(env.n_actions))
        for i in range(n_episodes):
            logging.info(f'Starting episode {i}')
            self._run_episode(env, action_value_estimates, model)
        return ActionValuePolicy(
            action_value_estimates,
            self._policy
        )
                
    def _run_episode(self, env: DiscreteEnvironment[ObsType, ActType], action_value_estimates: dict[ActType, np.ndarray], model: DynaModel):
        from_observation = env.reset(seed=42, return_info=False)
        is_done = False
        while not is_done:
            logging.debug(env.render('ansi'))
            best_action = self._policy(action_value_estimates[from_observation])
            next_observation, reward, is_done, _ = env.step(best_action)
            self._extend_model(model, from_observation, best_action, next_observation, reward, is_done)
            expected_reward = self._calculate_expected_reward(from_observation, best_action, next_observation, reward, action_value_estimates)
            action_value_estimates[from_observation][best_action] = expected_reward
            from_observation = next_observation
            self._planning(model, action_value_estimates)
        logging.debug('---------------------------------')

    def _extend_model(self, model: DynaModel, from_observation: ObsType, action: ActType, to_observation: ObsType, reward: float, is_done: bool):
        new_entry = DynaEntry(to_observation, reward, is_done)
        if from_observation not in model:
            model[from_observation] = { action : new_entry }
        else:
            model[from_observation][action] = new_entry

    def _sample_model(self, model: DynaModel) -> DynaMemory:
        observation = random.choice(list(model.keys()))
        action = random.choice(list(model[observation].keys()))
        entry = model[observation][action]
        return DynaMemory(observation, action, entry.next_observation, entry.reward, entry.is_done)

    def _planning(self, model: DynaModel, action_value_estimates: dict[ActType, np.ndarray]):
        for _ in range(self._planning_steps):
            from_observation, action, next_observation, reward, _ = self._sample_model(model)
            expected_reward = self._calculate_expected_reward(from_observation, action, next_observation, reward, action_value_estimates)
            action_value_estimates[from_observation][action] = expected_reward

    def _calculate_expected_reward(self, previous_observation: ObsType, action: ActType, 
                next_observation: ObsType, reward: float, action_value_estimates: dict[ObsType, np.ndarray[np._ShapeType, float]]) -> float:
        greedy_action_selection = GreedyActionSelection()
        best_next_action = greedy_action_selection(action_value_estimates[next_observation])
        td_target = reward + self._gamma * action_value_estimates[next_observation][best_next_action]
        td_delta = td_target - action_value_estimates[previous_observation][action]
        return action_value_estimates[previous_observation][action] + self._alpha * td_delta
