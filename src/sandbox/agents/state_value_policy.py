from dataclasses import dataclass
from typing import Generic
from gym.core import ObsType, ActType

@dataclass
class StateValuePolicy(Generic[ObsType, ActType]):
    state_value_estimates: dict[ObsType, float]
    _learned_actions: dict[ObsType, ActType]

    def select_action(self, from_observation: ObsType) -> ActType:
        return self._learned_actions[from_observation]
