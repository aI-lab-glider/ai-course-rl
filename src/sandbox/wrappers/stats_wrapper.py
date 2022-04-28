from dataclasses import astuple, dataclass, asdict
from typing import Optional, Tuple
import gym

@dataclass
class Statistic:
    cumulative_reward: float
    steps_count: int
    
    def increment(self, cumul_reward_increment, length_increment):
        return Statistic(
            self.cumulative_reward  + cumul_reward_increment,
            self.steps_count + length_increment
        )


STATS_KEY = 'episode_stats'

class StatsWrapper(gym.Wrapper):

    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.stats = []
        self._current_statistic = Statistic(0, 0)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self._current_statistic = self._current_statistic.increment(reward, 1)
        return observation, reward, done, {
            **info,
            STATS_KEY: asdict(self._current_statistic)
        }

    def reset(self, **kwargs):
        self.stats.append(self._current_statistic)
        self._current_statistic = Statistic(0, 0)
        return super().reset(**kwargs)
        
