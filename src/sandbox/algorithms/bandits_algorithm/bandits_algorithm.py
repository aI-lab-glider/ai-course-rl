from collections import defaultdict
from dataclasses import dataclass
from email.policy import default
from typing import TypeVar
from sandbox.action_selection_rules.generic import ActionSelectionRule
from sandbox.algorithms.algorithm import Algorithm
from gym.core import ObsType, ActType
import gym
from copy import deepcopy
import numpy as np

from sandbox.enviroments.multi_armed_bandit.env import BanditEnv
from sandbox.policies.bandit_policy import Bandit, BanditPolicy

ActionSelectionRuleType = TypeVar("ActionSelectionRuleType", bound=ActionSelectionRule)

class BanditsAlgorithm(Algorithm[None, int, ActionSelectionRuleType]):
    """
    An algorithm that allows testing how action selection rule behaves in particular scenario
    """
    
    def __init__(self, action_selection_rule: ActionSelectionRule) -> None:
        self._select_action = action_selection_rule

    def run(self, n_episodes: int, env: BanditEnv) -> ActionSelectionRuleType:
        select_action = deepcopy(self._select_action)
        bandits = [Bandit(0, 0) for _ in range(env.n_bandits)]
        for _ in range(n_episodes):
            self._run_episode(env, select_action, bandits)
        return BanditPolicy(
                bandits=bandits,
                _selection_rule=select_action
            )

    def _run_episode(self, env: BanditEnv, select_action: ActionSelectionRuleType, bandits: list[Bandit]):
        env.reset(seed=42)
        action = select_action(action_rewards=np.array([b.total_reward / max(b.call_count, 1) for b in bandits])) # Algorithm makes no assumptions about rewards
        _, reward, *_ = env.step(action)
        bandit = bandits[action]
        bandit.call_count += 1
        bandit.total_reward += reward




