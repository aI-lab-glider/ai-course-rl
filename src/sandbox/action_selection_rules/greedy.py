import random
from gym.core import ActType
import numpy as np

from sandbox.action_selection_rules.generic import ActionCandidate, ActionSelectionRule

class GreedyActionSelection(ActionSelectionRule[ActType]):
    def _select_action(self, action_rewards: list[ActionCandidate[ActType]]) -> ActType:
        rewards = np.array([a.reward for a in action_rewards])
        maxs = np.flatnonzero(rewards == np.max(rewards))
        best = random.choice(maxs)
        return action_rewards[best].action