import gym
from gym.core import ObsType, ActType


class NamedEnv(gym.Wrapper[ObsType, ActType]):
    def __init__(self, name: str, env: gym.Env[ObsType, ActType]):
        super().__init__(env)
        self.name = name

    def __repr__(self):
        return self.name
