from __future__ import annotations
from abc import ABC, abstractclassmethod, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import gym.spaces


class QNetwork(ABC):
    def __init__(
        self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete, learning_rate: float
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = self.build_model()

        self._loss_fn = keras.losses.mean_squared_error
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    @abstractmethod
    def build_model(self) -> Sequential:
        ...

    def copy(self) -> QNetwork:
        return keras.models.clone_model(self.model)

    def set_weights_from(self, other: QNetwork) -> None:
        self.model.set_weights(other.model.get_weights())

    def predict(self, observation):
        return self.model.predict(observation[np.newaxis])

    def update(self, states, mask, target_Q_values):
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



