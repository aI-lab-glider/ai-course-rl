from dataclasses import astuple, dataclass, asdict
from enum import Enum, IntEnum, auto
from typing import Iterable, Optional, Tuple
import gym
from matplotlib.axes import Axes
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

class PlotType(IntEnum):
    CumRewardvsTime = auto()
    EpisodeLengthvsTime = auto()
    EpisodeLengthHist = auto()
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
        
    def plot(self, types: PlotType = None):
        types = types or list(PlotType)
        cumulated_reward = [s.cumulative_reward for s in self.stats]
        steps_count = [s.steps_count for s in self.stats]
        ax = plt.subplots(figsize=(10, 10),
                               nrows=len(types),
                               ncols=1,
                               constrained_layout=True)[1]
        if not isinstance(ax, list):
            ax = [ax]

        for i, type in enumerate(types):
            match type:
                case PlotType.CumRewardvsTime:
                    ax[i].set_title("Episode Reward over Time")
                    ax[i].set_ylabel("Reward")
                    ax[i].set_xlabel("Time")
                    ax[i].grid(axis='y', which='major')
                    ax[i].plot(cumulated_reward,
                               '-',
                               c='orange',
                               linewidth=2)
                case PlotType.EpisodeLengthvsTime:
                    ax[i].set_title("Episode Length over Time")
                    ax[i].set_ylabel("Episode Length")
                    ax[i].set_xlabel("Time")
                    ax[i].grid(axis='y', which='major')
                    ax[i].plot(steps_count,
                               '-',
                               c='orange',
                               linewidth=2)
                case PlotType.EpisodeLengthHist:
                    ax[i].set_title("Episode per time step")
                    ax[i].set_xlabel("Episode Length")
                    ax[i].set_ylabel("Number of Episodes")
                    ax[i].hist(steps_count,
                               color='orange',
                               bins=len(self.stats))


