from dataclasses import astuple, dataclass, asdict
from typing import Optional, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np

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
        
    def plot(self):
        cumulated_reward = [s.cumulative_reward for s in self.stats]
        steps_count = [s.steps_count for s in self.stats]
        fig, ax = plt.subplots(figsize=(10, 10),
                               nrows=3,
                               ncols=1,
                               constrained_layout=True)
        ax[0].set_title("Episode Reward over Time")
        ax[0].set_ylabel("Reward")
        ax[0].set_xlabel("Time")
        ax[0].grid(axis='y', which='major')
        ax[0].plot(cumulated_reward,
                   '-',
                   c='orange',
                   linewidth=2)

        ax[1].set_title("Episode Length over Time")
        ax[1].set_ylabel("Episode Length")
        ax[1].set_xlabel("Time")
        ax[1].grid(axis='y', which='major')
        ax[1].plot(steps_count,
                   '-',
                   c='orange',
                   linewidth=2)

        ax[2].set_title("Episode per time step")
        ax[2].set_xlabel("Episode Length")
        ax[2].set_ylabel("Number of Episodes")
        ax[2].hist(steps_count,
                   color='orange',
                   bins=len(self.stats))

        fig.show()

