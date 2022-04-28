from dataclasses import dataclass
from typing import Generic, Protocol
from gym.core import ActType
import random

@dataclass
class ActionCandidate(Generic[ActType]):
    action: ActType
    reward: float


class Policy(Protocol, Generic[ActType]):
    def __call__(self, choices: list[ActionCandidate[ActType]]) -> ActType:
        ...

class GreedyPolicy(Generic[ActType]):
    def __call__(self, choices: list[ActionCandidate[ActType]]) -> ActType:
        return max(choices, key=lambda choice: choice.reward).action


class EpsilonGreedyPolicy(Generic[ActType]):
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon
        self._greedy_policy = GreedyPolicy[ActType]()

    def __call__(self, choices: list[ActionCandidate[ActType]]) -> ActType:
        best_action = self._greedy_policy(choices)
        if random.random() < self.epsilon:
            return best_action
        return random.choice([c for c in choices if c.action != best_action]).action

class EpsilonGreedyPolicyWithDecayEpsilon(Generic[ActType]):
    def __init__(self, initial_epsilon: float, decay_step: float) -> None:
        self.epsilon = initial_epsilon
        self.decay_step = decay_step
    
    def __call__(self, choices: list[ActionCandidate[ActType]]) -> ActType:
        epsilon_greedy_policy = EpsilonGreedyPolicy[ActType](self.epsilon)
        action = epsilon_greedy_policy(choices)
        self.epsilon *= self.decay_step
        return action
