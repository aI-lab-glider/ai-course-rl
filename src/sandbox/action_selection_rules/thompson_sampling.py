from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import math
from typing import Optional
from gym.core import ActType

from sandbox.action_selection_rules.generic import ActionCandidate, ActionSelectionRule
from sandbox.action_selection_rules.greedy import GreedyActionSelection


class ThompsonSampling(ActionSelectionRule[ActType]):
    def __init__(self):
        self._precisions = defaultdict[ActType, float](lambda: 0.001)
        self._n_calls = defaultdict[ActType, int](int)
        self._means = defaultdict[ActType, float](lambda: 5)
        self._avg_reward = defaultdict[ActType, float](int)
        self._select_action = GreedyActionSelection[ActType]()

    def _select_action(self, action_candidates: list[ActionCandidate[ActType]]) -> ActType:
        samples = [ActionCandidate(
            action=candidate.action,
            reward=self.sample(candidate.action),
        ) for candidate in action_candidates]

        action = self._select_action(samples)
        self.update(action, next(
            c.reward for c in action_candidates if c.action == action))
        return action

    def sample(self, action: ActType) -> float:
        return np.random.normal(
            self._means[action], 1 / np.sqrt(self._precisions[action]))

    def update(self, action: ActType, reward: float) -> None:
        prec = self._precisions[action]
        mean = self._means[action]
        self._means[action] = (prec*mean + self._n_calls[action] *
                               self._avg_reward[action]) / (prec + self._n_calls[action])
        self._precisions[action] += 1
        n = self.n_calls[action]
        self.n_calls[action] += 1
        self._avg_reward[action] = (n * self._avg_reward[action] + reward) / self.n_calls[
            action
        ]

        return super().update(action, reward)

    def __repr__(self) -> str:
        return type(self).__name__
