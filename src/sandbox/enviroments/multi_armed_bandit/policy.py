from abc import ABC, abstractmethod
import numpy as np
import math


class BanditPolicy(ABC):
    def __init__(self, n_bandits: int, init_value: float = 0.0) -> None:
        self.n_bandits = n_bandits
        self.avg_reward = [init_value] * self.n_bandits
        self.n_calls = [0] * self.n_bandits

    @abstractmethod
    def action(self) -> int:
        pass

    def update(self, action: int, reward: float) -> None:
        n = self.n_calls[action]
        self.n_calls[action] += 1
        self.avg_reward[action] = (n * self.avg_reward[action] + reward) / self.n_calls[
            action
        ]


class EpsilonGreedy(BanditPolicy):
    def __init__(self, n_bandits: int, eps: float = 0.1, init_value: float = 0.0):
        super().__init__(n_bandits, init_value)
        self.eps = eps

    def action(self) -> int:
        if np.random.random() < self.eps:
            return np.random.randint(0, self.n_bandits)
        return np.argmax(self.avg_reward)


class UCB(BanditPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = 0

    def action(self) -> int:
        ucb_scores = [
            self.ucb(r, a, self.t) for r, a in zip(self.avg_reward, self.n_calls)
        ]
        return np.argmax(ucb_scores)

    def update(self, action: int, reward: float) -> None:
        self.t += 1
        return super().update(action, reward)

    @staticmethod
    def ucb(
        mean_reward: float, action_counter: int, episode_number: int, c: int = 1.4
    ) -> float:
        if action_counter == 0:
            return float("inf")
        return mean_reward + c * (np.sqrt(math.log(episode_number) / action_counter))

