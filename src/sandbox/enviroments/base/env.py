from typing import Iterable, Optional
import gym
from gym.spaces.space import Space
from sandbox.enviroments.base.action import Action

from sandbox.enviroments.base.observation import Observation

class Environment(gym.Env):
    def __init__(self, all_observations, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._observations = all_observations

    def is_terminal_observation(self, observation: Observation) -> Optional[bool]:
        """Returns True if observation is a terminal state for the environment"""
        return None

    def action_space_for_observation(self, observation: Observation) -> Space[Action]:
        return self.action_space

    def all_observations(self) -> Iterable[Observation]:
        # TODO: fix me
        return self._observations