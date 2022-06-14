from __future__ import annotations
from abc import ABC, abstractmethod
import io
from typing import Collection, Iterable, Tuple, TypeVar, overload
from typing_extensions import Self
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import gym.spaces
import copy
from sandbox.action_selection_rules.greedy import GreedyActionSelection

from sandbox.policies.policy import Policy


class QNetwork(Policy):
    def __init__(
        self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete, learning_rate: float
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = self.build_model()
        self._loss_fn = keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    @abstractmethod
    def build_model(self) -> Sequential:
        ...

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def set_weights_from(self, other: QNetwork) -> None:
        self.model.set_weights(other.model.get_weights())

    @overload
    def predict(self, observation: np.ndarray[np._ShapeType, float]) -> np.ndarray[np._ShapeType, float]:
        """Prediction for a single observation

        Parameters
            observation: single observation to make prediction for
        Returns:
            Q values for every action from this observation
        """

    @overload
    def predict(self, observations: np.ndarray[np._ShapeType, np.ndarray]) -> np.ndarray[np._ShapeType, np.ndarray]:
        """Prediction for batch of observations
        Parameters
            observation: batch of observations
        Returns:
            batch of Q values for every action from every observations
        """

    def predict(self, observations):
        """Predicts Q-values for all actions that could be taken from given observation"""
        if isinstance(observations, Collection) and not isinstance(observations, np.ndarray):
            observations = np.array(observations)
        is_batch_prediction = observations.ndim >= 2
        # if single instance was passed, make it a batch
        obs = observations[np.newaxis] if not is_batch_prediction else observations
        predictions = self.model.predict(obs, verbose=0)
        return predictions if is_batch_prediction else predictions[0]

    def select_action(self, from_observation: np.ndarray[np._ShapeType, float]) -> np.ndarray[np._ShapeType, float]:
        greedy_selection = GreedyActionSelection()
        return greedy_selection(self.predict(from_observation))

    def update(self, observations: np.ndarray[np._ShapeType, np.ndarray], mask: np.ndarray[np._ShapeType, np.ndarray], target_Q_values: np.ndarray[np._ShapeType, np.ndarray]) -> None:
        """
        Gradient update

        Parameters
            observations: 2d numpy array representing batch of observations
            mask: 2d numpy array representing batch of one-hot encoded actions, that were actually taken
            target: 2d numpy array representing batch of target Q_values for the given observations
        returns 
            None

        """
        with tf.GradientTape() as tape:
            all_Q_values = self.model(observations)
            Q_values = tf.reduce_sum(
                all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

    def __repr__(self) -> str:
        with io.StringIO() as s:
            self.model.summary(print_fn=lambda x: s.write(x + '\n'))
            return f"QNetwork with {s.getvalue()}"
