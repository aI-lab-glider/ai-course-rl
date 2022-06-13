import gym
from gym.core import ObsType, ActType

class DiscreteEnvironment(gym.Wrapper[ObsType, ActType]):

    @property
    def n_actions(self):
        return self.env.action_space.n

    @property
    def n_observations(self):
        return self.env.observation_space.n

    def actions(self) -> list[ActType]:
        return [i for i in range(self.env.action_space.n)]
        
