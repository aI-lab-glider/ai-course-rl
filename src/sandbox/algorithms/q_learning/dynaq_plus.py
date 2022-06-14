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
from sandbox.algorithms.q_learning.dynaq import DynaEntry, DynaMemory, DynaModel

class DynaQPlus(Algorithm[ObsType, ActType, ActionValuePolicy]):

    def __init__(self, alpha: float, gamma: float, planning_steps: int, kappa: float, action_selection_rule: ActionSelectionRule[ActType]) -> None:
        self._alpha = alpha
        self._gamma = gamma
        self._action_selection_rule = action_selection_rule
        self._planning_steps = planning_steps
        self._kappa = kappa
        self._step = 0
        

    def run(self, n_episodes: int, env: DiscreteEnvironment[ObsType, ActType]):
        model: DynaModel = dict()
        tau = defaultdict(lambda: np.full(env.n_actions, self._step))
        action_value_estimates = defaultdict(lambda: np.zeros(env.n_actions))
        for i in range(n_episodes):
            logging.info(f'Starting episode {i}')
            self._run_episode(env, action_value_estimates, model, tau)
        return ActionValuePolicy(
            action_value_estimates,
            self._action_selection_rule
        )
                
    def _run_episode(self, env: DiscreteEnvironment[ObsType, ActType], action_value_estimates: dict[ActType, np.ndarray], model: DynaModel, tau: dict[ActType, np.ndarray]):
        from_observation = env.reset(seed=42, return_info=False)
        is_done = False
        while not is_done:
            logging.debug(env.render('ansi'))
            # TODO:
            # 1. the same as in DynaQ
            # 2. ...but:
            # - before updating the model, update tau 
            best_action = self._action_selection_rule(action_value_estimates[from_observation])
            next_observation, reward, is_done, _ = env.step(best_action)
            self._update_tau(tau, from_observation, best_action)
            self._extend_model(model, env, from_observation, best_action, next_observation, reward, is_done)
            expected_reward = self._calculate_expected_reward(from_observation, best_action, next_observation, reward, action_value_estimates)
            action_value_estimates[from_observation][best_action] = expected_reward
            from_observation = next_observation
            self._planning(model, tau, action_value_estimates)
        logging.debug('---------------------------------')

    def _update_tau(self, tau: dict[ActType, np.ndarray], observation: ObsType, action: ActType):
        # TODO:
        # - increcement number of steps (self._step) – this value is used to initialize new Tau values
        # - incremement all entries in tau
        # - tau for the observed action should set to 0
        self._step += 1
        for value in tau.values():
            value += 1
        tau[observation][action] = 0
    
    def _extend_model(self, model: DynaModel, env: DiscreteEnvironment[ObsType, ActType], from_observation: ObsType, action: ActType, to_observation: ObsType, reward: float, is_done: bool):
        # TODO:
        #   Almost the same as DynaQ, but when creating new dictionary, it has to cover all the actions (env.actions())
        #   the extra entries should have reward 0 and lead back to the same observation
        new_entry = DynaEntry(to_observation, reward, is_done)
        if from_observation not in model:
            model[from_observation] = { a : DynaEntry(from_observation, 0, False) for a in env.actions()}
        model[from_observation][action] = new_entry

    def _sample_model(self, model: DynaModel) -> DynaMemory:
        # TODO:
        # The same as in the DynaQ
        observation = random.choice(list(model.keys()))
        action = random.choice(list(model[observation].keys()))
        entry = model[observation][action]
        return DynaMemory(observation, action, entry.next_observation, entry.reward, entry.is_done)

    def _planning(self, model: DynaModel, tau: np.ndarray, action_value_estimates: dict[ActType, np.ndarray]):
        # TODO:
        # Almost thesame as Dyna, but reward should be alwaus increased by self._kappa * sqrt(tau) for the given memory 
        for _ in range(self._planning_steps):
            from_observation, action, next_observation, reward, _ = self._sample_model(model)
            reward += self._kappa * np.sqrt(tau[from_observation][action])
            expected_reward = self._calculate_expected_reward(from_observation, action, next_observation, reward, action_value_estimates)
            action_value_estimates[from_observation][action] = expected_reward

    def _calculate_expected_reward(self, previous_observation: ObsType, action: ActType, 
                next_observation: ObsType, reward: float, action_value_estimates: dict[ObsType, np.ndarray[np._ShapeType, float]]) -> float:
        # TODO:
        # Same as in vanilla QLearning
        greedy_action_selection = GreedyActionSelection()
        best_next_action = greedy_action_selection(action_value_estimates[next_observation])
        td_target = reward + self._gamma * action_value_estimates[next_observation][best_next_action]
        td_error = td_target - action_value_estimates[previous_observation][action]
        return action_value_estimates[previous_observation][action] + self._alpha * td_error
