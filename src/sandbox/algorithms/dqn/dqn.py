from __future__ import annotations
import logging
from sandbox.action_selection_rules.generic import ActionSelectionRule
from sandbox.policies.nn_policy import QNetwork
from sandbox.algorithms.algorithm import Algorithm
from sandbox.algorithms.dqn.replay_buffer import ReplayMemory, Transition


import gym
import gym.spaces
from typing import Callable, TypeVar
import numpy as np
import tensorflow as tf

from sandbox.wrappers.discrete_env_wrapper import DiscreteEnvironment


QNetworkType = TypeVar('QNetworkType', bound=QNetwork)
class DQNAlgorithm(Algorithm[np.ndarray, int, QNetworkType]):
    def __init__(
        self,
        memory_size: int,
        create_network: Callable[[gym.spaces.Box, gym.spaces.Discrete, float], QNetworkType],
        action_selection_rule: ActionSelectionRule[int],
        learning_rate: float,
        max_episode_length: int,
        exploration_steps: int,
        target_update_steps: int,
        batch_size: int,
        discount_rate: float,
    ):
        self.replay_buffer = ReplayMemory(memory_size)
        self.create_network = create_network
        self.learning_rate = learning_rate
        self.action_selection_rule = action_selection_rule
        self.max_episode_length = max_episode_length
        self.exploration_steps = exploration_steps
        self.target_update_steps = target_update_steps
        self.batch_size = batch_size
        self.discount_rate = discount_rate


    def run(self, n_episodes: int, env: DiscreteEnvironment[np.ndarray, int]) -> QNetworkType:
        self.env = env
        self.network = self.create_network(env.observation_space, env.action_space, self.learning_rate)
        self.target_network = self.network.copy()
        return self._train(
            n_episodes,
            self.max_episode_length,
            self.exploration_steps,
            self.target_update_steps,
            self.batch_size,
            self.discount_rate,
        )


    def _train(
        self,
        n_episodes: int,
        max_episode_length: int,
        exploration_steps: int,
        target_update_steps: int,
        batch_size: int,
        discount_rate: float,
    ) -> QNetworkType:
        self.discount_rate = discount_rate
        self.episodes_length = []

        for episode in range(n_episodes):
            logging.info(f"Starting episode: {episode}")
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

    def _play_one_step(self, observation: np.ndarray[np._ShapeType, float]) -> tuple[np.ndarray, float, bool, dict]:
        Q_values = self.network.predict(observation)
        action = self.action_selection_rule(Q_values)
        next_state, reward, done, info = self.env.step(action)
        self.replay_buffer.push(Transition(observation, action, next_state, reward, done))
        return next_state, reward, done, info

    def _training_step(self, batch_size: int) -> None:
        transitions = self.replay_buffer.sample(batch_size)
        observations, actions, next_states, rewards, dones = list(
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
        mask = tf.one_hot(actions, self.env.action_space.n).numpy()

        self.network.update(observations, mask, target_Q_values)
