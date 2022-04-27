from abc import ABC, abstractmethod
import numpy as np
import math
from typing import Optional


class BanditPolicy(ABC):
    def __init__(self, n_bandits: int, init_value: float = 0.0, name: Optional[str]=None) -> None:
        self.n_bandits = n_bandits
        self.avg_reward = [init_value] * self.n_bandits
        self.n_calls = [0] * self.n_bandits
        self.t = 1
        self._name = name or type(self).__name__

    @property 
    def name(self) -> str:
        return self._name

    @abstractmethod
    def action(self) -> int:
        pass

    def update(self, action: int, reward: float) -> None:
        n = self.n_calls[action]
        self.n_calls[action] += 1
        self.avg_reward[action] = (n * self.avg_reward[action] + reward) / self.n_calls[
            action
        ]
        self.t += 1


class EpsilonGreedy(BanditPolicy):
    def __init__(self, n_bandits: int, eps: float = 0.1, init_value: float = 0.0, name: Optional[str]=None):
        super().__init__(n_bandits, init_value, name=name)
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

    @staticmethod
    def ucb(
        mean_reward: float, action_counter: int, episode_number: int, c: int = 1.4
    ) -> float:
        if action_counter == 0:
            return float("inf")
        return mean_reward + c * (np.sqrt(math.log(episode_number) / action_counter))


class ThompsonSampling(BanditPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = 0
        self._precisions = np.full(self.n_bandits, 0.001)
        self._means = np.full(self.n_bandits, 5)

    def action(self) -> int:
        samples = list(map(self.sample, range(self.n_bandits)))
        return np.argmax(samples)

    def sample(self, bandit_idx) -> float:
        return np.random.normal(self._means[bandit_idx], 1 / np.sqrt(self._precisions[bandit_idx]))

    def update(self, action: int, reward: float) -> None:
        # update asssumes that the variance of the bandits = 1
        prec = self._precisions[action]
        mean = self._means[action]

        self._means[action] = (prec*mean + self.n_calls[action]*self.avg_reward[action]) / (prec + self.n_calls[action])
        self._precisions[action] += 1 
        return super().update(action, reward)