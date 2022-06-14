from dataclasses import dataclass
from typing import Generic, Tuple, overload
from gym.core import ActType
import random
import numpy as np
from abc import abstractmethod, ABC


@dataclass
class ActionCandidate(Generic[ActType]):
    action: ActType
    reward: float


class ActionSelectionRule(ABC, Generic[ActType]):
    @overload
    def __call__(self, action_rewards: list[ActionCandidate[ActType]]) -> ActType:
        ...

    @overload
    def __call__(self, action_rewards: np.ndarray[Tuple[int], float]) -> ActType:
        ...

    def __call__(self, action_rewards: list[ActionCandidate[ActType]] | np.ndarray) -> ActType:
        if isinstance(action_rewards, np.ndarray):
            action_rewards = [ActionCandidate(
                action, reward) for action, reward in enumerate(action_rewards)]
        return self._select_action(action_rewards)

    @abstractmethod
    def _select_action(self, action_rewards: list[ActionCandidate[ActType]]) -> ActType:
        ...
