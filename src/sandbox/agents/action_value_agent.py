from dataclasses import dataclass
from typing import Generic
import numpy as np
from gym.core import ObsType, ActType

from sandbox.policies.generic_policies import ActionCandidate, Policy

@dataclass
class ActionValueAgent(Generic[ObsType, ActType]):
    action_value_estimates: np.ndarray
    policy: Policy

    def select_action(self, from_observation: ObsType) -> ActType:
        return self.policy([ActionCandidate(a, self.action_value_estimates[from_observation, a]) for a in range(self.action_value_estimates.shape[1])])