from typing import Callable, TypeVar, Sequence, List
import itertools
from collections import defaultdict
import numpy as np
import gym

# q_new(s,a) = (1 - alpha) * q(s,q) + alpha * (R_t+1  + gamma * max_a'( q(s', a')))
# policy :: State -> Arr[foat] (probabilities for each action)

S = TypeVar("S")


class QLearning:
    def __init__(self, env: gym.Env, learning_rate: float = 0.6,
                 discount_factor: float = 1.0, epsilon: float = 0.1, episodes: int = 10*4):
        self.env = env
        self.learning_rate = learning_rate
        self.discout_factor = discount_factor
        self.epsilon = epsilon
        self.episodes = episodes
        self._policy_func = self._policy
        self.Q = defaultdict(lambda: np.zeros(shape=env.action_space.shape, dtype=float))
        self.stats = {"episode_len": np.zeros(self.episodes),
                      "episode_reward": np.zeros(self.episodes)
                      }

    # remove later?
    def _policy(self, state: S) -> List[float]:
        probabilities = np.ones(shape=self.env.action_space.shape,
                                dtype=float) * self.epsilon / np.sum(self.env.action_space.shape)

        best_action = np.argmax(self.Q[state])
        probabilities[best_action] += (1 - self.epsilon)
        return probabilities

    def _update_Q(self, action, state, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        temporal_difference = reward + self.discout_factor * self.Q[next_state][best_next_action]
        self.Q[state][action] *= (1 - self.learning_rate)
        self.Q[state][action] += self.learning_rate * temporal_difference

    def _update_stats(self, episode, step, reward):
        self.stats["episode_len"][episode] = step
        self.stats["episode_reward"][episode] += reward

    def learn(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            for step in itertools.count():
                probabilities = self._policy_func(state)
                action = np.random.choice(np.arange(
                    len(probabilities), p=probabilities))

                next_state, reward, done, _ = self.env.step(action)
                self._update_stats(episode, step, reward)
                self._update_Q(action, state, reward, next_state)

                if done:
                    break

                state = next_state


