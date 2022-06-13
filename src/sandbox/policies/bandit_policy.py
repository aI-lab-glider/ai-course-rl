from dataclasses import dataclass

import numpy as np

from sandbox.action_selection_rules.generic import ActionSelectionRule


@dataclass
class Bandit:
    total_reward: float
    call_count: int

@dataclass
class BanditPolicy:
    bandits: Bandit
    _selection_rule: ActionSelectionRule

    def select_action(self):
        mean_rewards = np.array([b.total_reward / max(b.call_count, 1) for b in self.bandits])
        return self._selection_rule(mean_rewards)
