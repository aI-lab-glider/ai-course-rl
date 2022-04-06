from abc import ABC, abstractmethod 
import numpy as np 


class BanditPolicy(ABC):

    def __init__(self, n_bandits: int) -> None:
        self.n_bandits = n_bandits

    @abstractmethod
    def action(self) -> int:
        pass 

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        pass 


class EpsilonGreedy(BanditPolicy):

    def __init__(self, n_bandits, eps: float=0.1, init_value: float=0.):
        super().__init__(n_bandits)
        self.avg_reward = [init_value] * self.n_bandits
        self.n_calls = [0] * self.n_bandits
        self.eps = eps

    def action(self) -> int:
        if np.random.random() < self.eps:
            return np.random.randint(0, self.n_bandits)
        return np.argmax(self.avg_reward) 

    def update(self, action: int, reward: float) -> None:
        n = self.n_calls[action]
        self.n_calls[action] += 1
        self.avg_reward[action] = (n*self.avg_reward[action] + reward) / self.n_calls[action]
        
