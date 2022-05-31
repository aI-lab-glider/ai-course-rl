from gym.core import ActType
import random
from sandbox.action_selection_rules.epsilon_greedy import EpsilonGreedyActionSelection

from sandbox.action_selection_rules.generic import ActionCandidate, ActionSelectionRule

class EpsilonGreedyActionSelectionWithDecayEpsilon(ActionSelectionRule[ActType]):
    def __init__(self, initial_epsilon: float, decay_step: float) -> None:
        self.epsilon = initial_epsilon
        self.decay_step = decay_step
    
    def _select_action(self, action_rewards: list[ActionCandidate[ActType]]) -> ActType:
        epsilon_greedy_policy = EpsilonGreedyActionSelection[ActType](self.epsilon)
        action = epsilon_greedy_policy(action_rewards)
        self.epsilon *= self.decay_step
        return action
