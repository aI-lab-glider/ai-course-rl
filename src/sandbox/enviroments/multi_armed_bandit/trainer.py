from sandbox.enviroments.multi_armed_bandit.env import BanditEnv
from sandbox.enviroments.multi_armed_bandit.history import History
from sandbox.enviroments.multi_armed_bandit.policy import BanditPolicy
import matplotlib.pyplot as plt
from typing import Union
import numpy as np


class BanditTrainer:
    def __init__(self, env: BanditEnv, policy: Union[BanditPolicy, list[BanditPolicy]]):
        self._env = env
        self._policies = policy if isinstance(policy, list) else [policy]
        self.history = History(self._env, self._policies)

    def train(self, n_episodes: int) -> list[BanditPolicy]:
        _ = self._env.reset()
        for _ in range(n_episodes):
            actions = []
            rewards = []
            for p in self._policies:
                action = p.action()
                _, reward, _, _ = self._env.step(action)
                actions.append(action)
                rewards.append(reward)
                p.update(action, reward)

            self.history.update(actions, rewards)
        return self._policies
