from dataclasses import dataclass
from typing import Generic
import numpy as np
from gym.core import ObsType, ActType

from sandbox.action_selection_rules.generic import ActionCandidate, ActionSelectionRule

@dataclass
class ActionValuePolicy(Generic[ObsType, ActType]):
    action_value_estimates: np.ndarray
    selection_rule: ActionSelectionRule

    def select_action(self, from_observation: ObsType) -> ActType:
        return self.selection_rule([ActionCandidate(a, self.action_value_estimates[from_observation, a]) for a in range(self.action_value_estimates.shape[1])])