from sandbox.enviroments.multi_armed_bandit.env import BanditEnv
from sandbox.enviroments.multi_armed_bandit.policy import BanditPolicy
import matplotlib.pyplot as plt 
import numpy as np 

class BanditTrainer:
    def __init__(self, env: BanditEnv, policy: BanditPolicy):
        self._env = env 
        self._policy = policy 

        self.action_history = []
        self.reward_history = []

    def train(self, n_episodes: int) -> BanditPolicy:
        _ = self._env.reset()
        for _ in range(n_episodes):
            action = self._policy.action()
            _, reward, _, _ = self._env.step(action)
            self._policy.update(action, reward)

            self.action_history.append(action)
            self.reward_history.append(reward)
        return self._policy

    def display_history(self) -> None:
        n = len(self.reward_history)
        avg_reward = np.cumsum(self.reward_history) / np.arange(1, n + 1)

        plt.subplot(1, 2, 1)
        plt.xlabel("Steps")
        plt.ylabel("Average reward")
        plt.plot(range(n), avg_reward)

        distr = self._env.distributions
        optimal_action = max(range(len(distr)), key=lambda d: distr[d].mean)
        optimal_indicator = np.array(self.action_history) == optimal_action
        pr_optimal_action = np.cumsum(optimal_indicator) / np.arange(1, n + 1)

        plt.subplot(1, 2, 2)
        plt.xlabel("Steps")
        plt.ylabel("% Optimal action")
        plt.plot(range(n), pr_optimal_action)

        plt.show()