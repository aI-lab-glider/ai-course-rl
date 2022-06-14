from dataclasses import astuple, dataclass, asdict
from enum import Enum, IntEnum, auto
from itertools import accumulate
import logging
from typing import Iterable, Optional, Tuple
import gym
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Statistic:
    episode_reward: float
    steps_count: int

    def increment(self, step_reward, length_increment):
        return Statistic(
            self.episode_reward + step_reward,
            self.steps_count + length_increment
        )


STATS_KEY = 'episode_stats'


class PlotType(IntEnum):
    RewardsVsEpNumber = auto()
    EpisodeLengthvsStepsCount = auto()
    EpisodeLengthHist = auto()
    CumulatedReward = auto()


class StatsWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, enable_logging=True):
        super().__init__(env)
        self.stats: list[Statistic] = []
        self._current_statistic = Statistic(0, 0)
        self._enable_logging = enable_logging
        self.steps_count = 0

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self._current_statistic = self._current_statistic.increment(reward, 1)
        self.steps_count += 1
        return observation, reward, done, {
            **info,
            STATS_KEY: asdict(self._current_statistic)
        }

    def reset(self, **kwargs):
        self.stats.append(self._current_statistic)
        if self._enable_logging:
            logging.info(f'Episode stats: {self._current_statistic}')
        self._current_statistic = Statistic(0, 0)
        self.steps_count = 0
        return super().reset(**kwargs)

    def plot(self, types: PlotType = None, ax: list[Axes] = None, color=None):
        types = types or list(PlotType)
        episode_rewards = [s.episode_reward for s in self.stats]
        steps_count = [s.steps_count for s in self.stats]
        if ax is None:
            ax = plt.subplots(figsize=(10, 10),
                              nrows=len(types),
                              ncols=1,
                              constrained_layout=True, squeeze=False)[1]
        title_prefix = f"{self.env.name}: " if hasattr(
            self.env, 'name') else ''

        for i, type in enumerate(types):
            match type:
                case PlotType.RewardsVsEpNumber:
                    ax[i].set_title(
                        f"{title_prefix}Episode Reward vs Steps Count")
                    ax[i].set_ylabel("Reward")
                    ax[i].grid(axis='y', which='major')
                    ax[i].plot(episode_rewards,
                               '-',
                               c=color or 'orange',
                               linewidth=2)
                case PlotType.EpisodeLengthvsStepsCount:
                    ax[i].set_title(
                        f"{title_prefix}Episode Length vs Steps Count")
                    ax[i].set_ylabel("Episode Length")
                    ax[i].grid(axis='y', which='major')
                    ax[i].plot(steps_count,
                               '-',
                               c=color or 'orange',
                               linewidth=2)
                case PlotType.EpisodeLengthHist:
                    ax[i].set_title(f"{title_prefix}Episode per steps count")
                    ax[i].set_ylabel("Number of Episodes")
                    ax[i].hist(steps_count,
                               color=color or 'orange',
                               bins=len(self.stats))
                case PlotType.CumulatedReward:
                    ax[i].set_title(
                        f"{title_prefix}Cumulated reward vs Steps Count")
                    ax[i].set_ylabel("Reward")
                    ax[i].grid(axis='y', which='major')
                    ax[i].plot(list(accumulate(episode_rewards)),
                               '-',
                               c=color or 'orange',
                               linewidth=2)

    def average_reward(self):
        return sum([s.episode_reward for s in self.stats]) / len(self.stats)

    def average_step_count(self):
        return sum([s.steps_count for s in self.stats]) / len(self.stats)
