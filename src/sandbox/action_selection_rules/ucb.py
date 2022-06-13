from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import math
from typing import Optional
from gym.core import ActType
from sandbox.action_selection_rules.generic import ActionCandidate, ActionSelectionRule
from sandbox.action_selection_rules.greedy import GreedyActionSelection

@dataclass
class UCB(ActionSelectionRule[ActType]):
    c: float

    def __init__(self, c: float = 1.414):
        self.c = c
        self._n_calls = defaultdict[ActType, int](int)
        self.select_action = GreedyActionSelection[ActType]()


    def _select_action(self, action_candidates: list[ActionCandidate[ActType]]) -> ActType:
        """
        Expects action cadidates to be dependent on state from which this actions are performed.
        """
        t = sum(self._n_calls.values())
        ucb_scores = [
            ActionCandidate(
                action=candidate.action,
                reward=self._ucb(candidate.reward, self._n_calls[candidate.action], t, self.c)
            ) for candidate in action_candidates
        ]
        action = self.select_action(ucb_scores)
        self._n_calls[action] += 1
        return action
    
    @staticmethod
    def _ucb(
        mean_reward: float, action_counter: int, episode_number: int, c: int = 1.414
    ) -> float:
        if action_counter == 0:
            return float("inf")
        return mean_reward + c * (np.sqrt(math.log(episode_number) / action_counter))