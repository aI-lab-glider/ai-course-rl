from gym.core import ActType

from sandbox.action_selection_rules.generic import ActionCandidate, ActionSelectionRule

class GreedyActionSelection(ActionSelectionRule[ActType]):
    def _select_action(self, action_rewards: list[ActionCandidate[ActType]]) -> ActType:
        return max(action_rewards, key=lambda action_candidate: action_candidate.reward).action