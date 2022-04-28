from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar, Typing, Generic
import numpy as np
import gym
from sandbox.enviroments.multi_armed_bandit.policy import Policy

TValueEstimates = TypeVar('TValueEstimates')

class Algorithm(ABC, Generic[TValueEstimates]):

    @abstractmethod
    def run(self, n_episodes: int, evniroment: gym.Env, policy: Policy) -> TValueEstimates:
        ...
        