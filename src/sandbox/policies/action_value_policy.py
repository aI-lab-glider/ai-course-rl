from dataclasses import dataclass
from typing import Generic
import numpy as np
from gym.core import ObsType, ActType

from sandbox.action_selection_rules.generic import ActionCandidate, ActionSelectionRule


@dataclass
class ActionValuePolicy(Generic[ObsType, ActType]):
    action_value_estimates: dict[ObsType, np.ndarray]
    selection_rule: ActionSelectionRule

    def select_action(self, from_observation: ObsType) -> ActType:
        return self.selection_rule(self.action_value_estimates[from_observation])

    def __repr__(self) -> str:
        return f'{type(self).__name__} with {repr(self.selection_rule)}'
