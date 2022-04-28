from abc import ABC, abstractmethod
import gym
from typing import Generic, TypeVar
from gym.core import ObsType, ActType


AlgorithmResult = TypeVar("AlgorithmResult")
class Algorithm(Generic[ObsType, ActType, AlgorithmResult],ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def run(self, n_episodes: int, evn: gym.Env[ObsType, ActType]) -> AlgorithmResult:
        ...
        