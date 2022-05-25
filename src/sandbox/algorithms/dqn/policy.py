from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import gym.spaces, copy 


class QNetwork(ABC):
    def __init__(
        self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete, learning_rate: float
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = self.build_model()

        self._loss_fn = keras.losses.Hubert()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    @abstractmethod
    def build_model(self) -> Sequential:
        ...

    def copy(self) -> QNetwork:
        return copy.deepcopy(self)

    def set_weights_from(self, other: QNetwork) -> None:
        self.model.set_weights(other.model.get_weights())

    def predict(self, observation):
        obs = observation[np.newaxis] if observation.ndim < 2 else observation # if vector was given, make it an matrix
        return self.model.predict(obs)

    def update(self, states: np.ndarray, mask: np.ndarray, target_Q_values: np.ndarray) -> None:
        """
        Gradient update

        Parameters
            states: 2d numpy array representing batch of observations
            mask: 2d numpy array representing batch of one-hot encoded actions, that were actually taken
            target: 2d numpy array representing batch of target Q_values for the given states
        returns 
            None

        """
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


class MyQNetwork(QNetwork):
    def build_model(self) -> Sequential:
        model = Sequential(
            [
                keras.layers.Dense(32, activation="elu", input_shape=self.observation_space.high.shape),
                keras.layers.Dense(32, activation="elu"),
                keras.layers.Dense(self.action_space.n),
            ]
        )
        return model


def epsilon_greedy_policy(
    eps, state, network: QNetwork, action_space: gym.spaces.Space
) -> int:
    if np.random.rand() < eps:
        return action_space.sample()
    Q_values = network.predict(state)
    return int(np.argmax(Q_values))
