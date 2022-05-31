from abc import ABC, abstractmethod
import gym
from typing import Generic, TypeVar
from gym.core import ObsType, ActType


PolicyType = TypeVar("PolicyType")
class Algorithm(Generic[ObsType, ActType, PolicyType],ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def run(self, n_episodes: int, evn: gym.Env[ObsType, ActType]) -> PolicyType:
        ...
        