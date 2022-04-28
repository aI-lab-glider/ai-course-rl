from abc import ABC, abstractmethod
from typing import Hashable

from sandbox.algorithms.algorithm import Action, Observation


class StateValueEstimates(ABC):
    """An array with estimates of state-value function or action-value"""
    @abstractmethod
    def get(self, observation: Observation) -> float:
        ...

    @abstractmethod
    def update(self, observation: Observation, value: float) -> None:
        ...
    
    @abstractmethod
    def __contains__(self, key):
        ...


class ActionValueEstimates(ABC):
    """An array with estimates of state-value function or action-value"""
    @abstractmethod
    def get(self, observation_action: tuple[Observation, Action]) -> float:
        ...

    @abstractmethod
    def update(self, observation, action, value: float) -> None:
        ...

    @abstractmethod
    def __contains__(self, key):
        ...


