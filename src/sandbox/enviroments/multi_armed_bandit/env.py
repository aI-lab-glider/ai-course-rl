import gym
import numpy as np 
from collections import namedtuple


NormalDistribution = namedtuple("NormalDistribution", "mean std")



class BanditEnv(gym.Env[int, None]):

    def __init__(self, reward_distributions: list[NormalDistribution]):
        self.distributions = reward_distributions.copy()
        self.n_bandits = len(reward_distributions)

        self.action_space = gym.spaces.Discrete(self.n_bandits)
        self.observation_space = gym.spaces.Discrete(1) 
        self._observation = None # bandit env has no observation, but has to return something

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Unsupported action {action} for action space {self.action_space}")
        
        bandit = self.distributions[action]
        reward = np.random.normal(bandit.mean, bandit.std)
        done = False 
        info = {}
        return self._observation, reward, done, info

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        return self._observation
