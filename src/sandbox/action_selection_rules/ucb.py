from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import math
from typing import Optional
from gym.core import ActType
from sandbox.action_selection_rules.generic import ActionCandidate, ActionSelectionRule
from sandbox.action_selection_rules.greedy import GreedyActionSelection


class UCB(ActionSelectionRule[ActType]):
    def __init__(self):
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
                reward=self._ucb(candidate.reward, self._n_calls[candidate.action], t)
            ) for candidate in action_candidates
        ]
        action = self.select_action(ucb_scores)
        self._n_calls[action] += 1
        return action
    
    @staticmethod
    def _ucb(
        mean_reward: float, action_counter: int, episode_number: int, c: int = 1.4
    ) -> float:
        if action_counter == 0:
            return float("inf")
        return mean_reward + c * (np.sqrt(math.log(episode_number) / action_counter))