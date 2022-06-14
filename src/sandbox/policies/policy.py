from abc import ABC, abstractmethod
from gym.core import ObsType, ActType
from typing import Generic


class Policy(ABC, Generic[ObsType, ActType]):
    @abstractmethod
    def select_action(self, from_observation: ObsType) -> ActType:
        ...
