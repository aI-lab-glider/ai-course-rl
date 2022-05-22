from dataclasses import dataclass
from typing import Generic, Protocol, Tuple, overload
from gym.core import ActType
import random
import numpy as np
from abc import abstractmethod, ABC

@dataclass
class ActionCandidate(Generic[ActType]):
    action: ActType
    reward: float


class Policy(ABC, Generic[ActType]):
    @overload
    def __call__(self, action_rewards: list[ActionCandidate[ActType]]) -> ActType:
        ...

    @overload
    def __call__(self, action_rewards: np.ndarray[Tuple[int], float]) -> ActType:
        ...
    
    def __call__(self, action_rewards: list[ActionCandidate[ActType]] | np.ndarray) -> ActType:
        if isinstance(action_rewards, np.ndarray):
            action_rewards = [ActionCandidate(action, reward) for action, reward in enumerate(action_rewards)]
        return self._select_action(action_rewards)
    
    @abstractmethod
    def _select_action(self, action_rewards: list[ActionCandidate[ActType]]) -> ActType:
        ...

class GreedyPolicy(Policy[ActType]):
    def _select_action(self, action_rewards: list[ActionCandidate[ActType]]) -> ActType:
        return max(action_rewards, key=lambda action_candidate: action_candidate.reward).action


class EpsilonGreedyPolicy(Policy[ActType]):
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon
        self._greedy_policy = GreedyPolicy[ActType]()

    def _select_action(self, action_rewards: list[ActionCandidate[ActType]]) -> ActType:
        best_action = self._greedy_policy(action_rewards)
        if random.random() > self.epsilon:
            return best_action
        return random.choice([c for c in action_rewards if c.action != best_action]).action

class EpsilonGreedyPolicyWithDecayEpsilon(Policy[ActType]):
    def __init__(self, initial_epsilon: float, decay_step: float) -> None:
        self.epsilon = initial_epsilon
        self.decay_step = decay_step
    
    def _select_action(self, action_rewards: list[ActionCandidate[ActType]]) -> ActType:
        epsilon_greedy_policy = EpsilonGreedyPolicy[ActType](self.epsilon)
        action = epsilon_greedy_policy(action_rewards)
        self.epsilon *= self.decay_step
        return action
