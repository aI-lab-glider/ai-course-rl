from dataclasses import dataclass
from gym.core import ActType
import random
from sandbox.action_selection_rules.epsilon_greedy import EpsilonGreedyActionSelection

from sandbox.action_selection_rules.generic import ActionCandidate, ActionSelectionRule

@dataclass
class EpsilonGreedyActionSelectionWithDecayEpsilon(ActionSelectionRule[ActType]):
    epsilon: float
    decay_step: float
    
    def _select_action(self, action_rewards: list[ActionCandidate[ActType]]) -> ActType:
        epsilon_greedy_action_selection = EpsilonGreedyActionSelection[ActType](self.epsilon)
        action = epsilon_greedy_action_selection(action_rewards)
        self.epsilon *= self.decay_step
        return action
    