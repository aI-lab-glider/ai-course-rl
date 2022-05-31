from gym.core import ActType
import random

from sandbox.action_selection_rules.generic import ActionCandidate, ActionSelectionRule
from sandbox.action_selection_rules.greedy import GreedyActionSelection

class EpsilonGreedyActionSelection(ActionSelectionRule[ActType]):
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon
        self._greedy_policy = GreedyActionSelection[ActType]()

    def _select_action(self, action_rewards: list[ActionCandidate[ActType]]) -> ActType:
        best_action = self._greedy_policy(action_rewards)
        if random.random() > self.epsilon:
            return best_action
        return random.choice([c for c in action_rewards if c.action != best_action]).action