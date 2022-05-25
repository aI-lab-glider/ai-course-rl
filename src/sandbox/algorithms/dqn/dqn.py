from sandbox.algorithms.dqn.replay_buffer import ReplayMemory, Transition
from sandbox.algorithms.dqn.policy import MyQNetwork, QNetwork, epsilon_greedy_policy

import gym
from abc import ABC
from typing import Callable
import gym.spaces
import numpy as np
import tensorflow as tf


class DQNAlgorithm:
    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        network: type[QNetwork],
        policy: Callable,
        learning_rate: float,
    ):
        self.env = env
        self.replay_buffer = ReplayMemory(memory_size)
        self.network = network(env.observation_space, env.action_space, learning_rate)
        self.target_network = self.network.copy()
        self.policy = policy

    def train(
        self,
        n_episodes: int,
        max_episode_length: int,
        exploration_steps: int,
        target_update_steps: int,
        batch_size: int,
        discount_rate: float,
    ):
        self.discount_rate = discount_rate
        self.episodes_length = []

        for episode in range(n_episodes):
            self._rollout(max_episode_length)
            if len(self.replay_buffer) <= exploration_steps:
                continue

            self._training_step(batch_size)
            if episode % target_update_steps == 0:
                self.target_network.set_weights_from(self.network)
        return self.network

    def _rollout(self, max_episode_length: int):
        obs = self.env.reset()
        for t in range(max_episode_length):
            obs, reward, done, info = self._play_one_step(obs)
            if done:
                break
        self.episodes_length.append(t)

    def _play_one_step(self, state) -> tuple[np.ndarray, float, bool, dict]:
        action = self.policy(state, self.network, self.env.action_space)
        next_state, reward, done, info = self.env.step(action)
        self.replay_buffer.push(Transition(state, action, next_state, reward, done))
        return next_state, reward, done, info

    def _training_step(self, batch_size: int) -> None:
        transitions = self.replay_buffer.sample(batch_size)
        states, actions, next_states, rewards, dones = list(
            map(np.array, zip(*transitions))
        )
        next_Q_values = self.network.predict(next_states)
        best_next_actions = np.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self.env.action_space.n).numpy()
        
        next_target_Q_values = (
            self.target_network.predict(next_states) * next_mask
        ).sum(axis=1)

        target_Q_values = (
            rewards + (1 - dones) * self.discount_rate * next_target_Q_values
        )
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.env.action_space.n)

        self.network.update(states, mask, target_Q_values)


if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    dqn = DQNAlgorithm(
        env,
        memory_size=2000,
        network=MyQNetwork,
        policy=lambda *args: epsilon_greedy_policy(0.01, *args),
        learning_rate=1e-4,
    )
    dqn.train(6, 200, 50, 50)
