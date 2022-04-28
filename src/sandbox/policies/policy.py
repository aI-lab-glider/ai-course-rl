
from abc import ABC, abstractmethod
from typing import Generic, Iterable, List, TypeVar, overload

from sandbox.enviroments.base.observation import Observation
from sandbox.value_estimates.value_estimates import Action, ActionValueEstimates, StateValueEstimates, ValueEstimates




TActionValueEstimates = TypeVar(
    'TActionValueEstimates', bound=ActionValueEstimates)

class ActionValuePolicy(ABC, Generic[TActionValueEstimates]):
    def __init__(self, value_estimates: TActionValueEstimates) -> None:
        self._action_value_estimates = value_estimates

    @abstractmethod
    def get(self, observation: Observation, actions: list[Action]) -> Action:
        ...


TStateValueEstimates = TypeVar(
    'TStateValueEstimates', bound=StateValueEstimates)


class StateValuePolicy(ABC, Generic[TStateValueEstimates]):
    def __init__(self, value_estimates: TStateValueEstimates) -> None:
        self._value_estimates = value_estimates

    @abstractmethod
    def get(self, observations: list[Observation]) -> Action:
        ...
