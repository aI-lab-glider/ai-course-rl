from dataclasses import dataclass
from types import NoneType

import numpy as np

from sandbox.action_selection_rules.generic import ActionSelectionRule
from sandbox.policies.policy import Policy


@dataclass
class Bandit:
    total_reward: float
    call_count: int


@dataclass
class BanditPolicy(Policy[NoneType, int]):
    bandits: Bandit
    _selection_rule: ActionSelectionRule[int]

    def select_action(self, observation: NoneType) -> int:
        assert observation is None, "Bandit policy does not depend on current observation"
        mean_rewards = np.array(
            [b.total_reward / max(b.call_count, 1) for b in self.bandits])
        return self._selection_rule(mean_rewards)

    def __repr__(self) -> str:
        return f'{type(self).__name__} with {repr(self._selection_rule)}'
