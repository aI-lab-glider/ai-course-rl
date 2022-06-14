import gym
from gym.core import ObsType, ActType


class DiscreteEnvironment(gym.Wrapper[ObsType, ActType]):

    def __init__(self, env: gym.Env):
        assert hasattr(env.action_space, 'n') and hasattr(
            env.action_space, 'n')
        super().__init__(env)

    @property
    def n_actions(self):
        return self.env.action_space.n

    @property
    def n_observations(self):
        return self.env.observation_space.n

    def actions(self) -> list[ActType]:
        return [i for i in range(self.env.action_space.n)]
