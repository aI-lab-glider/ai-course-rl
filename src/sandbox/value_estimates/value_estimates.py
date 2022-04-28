from abc import ABC, abstractmethod
from typing import Hashable

from sandbox.algorithms.algorithm import Action, Observation


class StateValueEstimates(ABC):
    """An array with estimates of state-value function or action-value"""
    @abstractmethod
    def __getitem__(self, observation: Observation) -> float:
        ...

    @abstractmethod
    def __setitem__(self, observation: Observation, value: float) -> None:
        ...


class ActionValueEstimates(ABC):
    """An array with estimates of state-value function or action-value"""
    @abstractmethod
    def __getitem__(self, observation_action: tuple[Observation, Action]) -> float:
        ...

    @abstractmethod
    def __setitem__(self, observation: tuple[Observation, Action], value: float) -> None:
        ...

