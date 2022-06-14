from dataclasses import dataclass
from gym.core import ActType
import random

from sandbox.action_selection_rules.generic import ActionCandidate, ActionSelectionRule
from sandbox.action_selection_rules.greedy import GreedyActionSelection


@dataclass
class EpsilonGreedyActionSelection(ActionSelectionRule[ActType]):
    epsilon: float

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon
        self._greedy_policy = GreedyActionSelection[ActType]()

    def _select_action(self, action_rewards: list[ActionCandidate[ActType]]) -> ActType:
        if random.random() > self.epsilon:
            return self._greedy_policy(action_rewards)
        return random.choice([c for c in action_rewards]).action