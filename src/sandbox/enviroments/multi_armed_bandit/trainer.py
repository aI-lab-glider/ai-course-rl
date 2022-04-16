from sandbox.enviroments.multi_armed_bandit.env import BanditEnv
from sandbox.enviroments.multi_armed_bandit.policy import BanditPolicy
import matplotlib.pyplot as plt
from typing import Union
import numpy as np


class BanditTrainer:
    def __init__(self, env: BanditEnv, policy: Union[BanditPolicy, list[BanditPolicy]]):
        self._env = env
        self._policies = policy if isinstance(policy, list) else [policy]

        self.action_history = []
        self.reward_history = []

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

            self.action_history.append(actions)
            self.reward_history.append(rewards)
        return self._policies

    def display_history(self) -> None:
        n = len(self.reward_history)
        policies_avg_reward = np.cumsum(self.reward_history, axis=0) / np.array([np.arange(1, n + 1)]).T
        print(policies_avg_reward.shape)

        plt.subplot(1, 2, 1)
        plt.xlabel("Steps")
        plt.ylabel("Average reward")
        for p, avg_reward in zip(self._policies, policies_avg_reward.T):
            plt.plot(range(n), avg_reward, label=type(p).__name__)

        distr = self._env.distributions
        optimal_action = max(range(len(distr)), key=lambda d: distr[d].mean)
        optimal_indicator = np.array(self.action_history) == optimal_action
        pr_optimal_action = (
            np.cumsum(optimal_indicator, axis=0) / np.array([np.arange(1, n + 1)]).T
        )

        plt.subplot(1, 2, 2)
        plt.xlabel("Steps")
        plt.ylabel("% Optimal action")
        for p, optimal_action_arr in zip(self._policies, pr_optimal_action.T):
            plt.plot(range(n), optimal_action_arr, label=type(p).__name__)

        plt.legend()
        plt.show()
