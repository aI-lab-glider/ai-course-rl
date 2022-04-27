from sandbox.enviroments.multi_armed_bandit.env import BanditEnv
from sandbox.enviroments.multi_armed_bandit.policy import BanditPolicy
import matplotlib.pyplot as plt
import numpy as np 


class History:
    def __init__(self, env: BanditEnv, policies: list[BanditPolicy]):
        self._policies = policies
        self._env = env
        self._action_history: list[list[int]] = []
        self._reward_history: list[list[float]] = []
    
    def update(self, actions: list[int], rewards: list[float]) -> None:
        assert len(actions) == len(self._policies), f"Invalid lengh of action list"
        assert len(rewards) == len(self._policies), f"Invalid lengh of reward list"

        self._action_history.append(actions)
        self._reward_history.append(rewards)

    def display(self) -> None:
        plots = [self._display_avg_reward, self._display_optimal_actions]

        for i, method in enumerate(plots):
            plt.subplot(1, len(plots), i + 1)
            method.__call__()

        plt.legend()
        plt.show()

    def _display_avg_reward(self) -> None:
        n = len(self._reward_history)
        policies_avg_reward = np.cumsum(self._reward_history, axis=0) / np.array([np.arange(1, n + 1)]).T

        plt.xlabel("Steps")
        plt.ylabel("Average reward")
        for p, avg_reward in zip(self._policies, policies_avg_reward.T):
            plt.plot(range(n), avg_reward, label=p.name)

    def _display_optimal_actions(self) -> None:
        n = len(self._reward_history)
        distr = self._env.distributions
        optimal_action = max(range(len(distr)), key=lambda d: distr[d].mean)
        optimal_indicator = np.array(self._action_history) == optimal_action
        pr_optimal_action = (
            np.cumsum(optimal_indicator, axis=0) / np.array([np.arange(1, n + 1)]).T
        )

        plt.xlabel("Steps")
        plt.ylabel("% Optimal action")
        for p, optimal_action_arr in zip(self._policies, pr_optimal_action.T):
            plt.plot(range(n), optimal_action_arr, label=p.name)