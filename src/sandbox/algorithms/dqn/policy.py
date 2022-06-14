from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, overload
from typing_extensions import Self
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import gym.spaces
import copy

from sandbox.policies.nn_policy import QNetwork


class CartPoleQNetwork(QNetwork):
    def build_model(self) -> Sequential:
        model = Sequential(
            [
                keras.layers.Dense(
                    32, activation="elu", input_shape=self.observation_space.high.shape),
                keras.layers.Dense(32, activation="elu"),
                keras.layers.Dense(self.action_space.n, activation="softmax")
            ]
        )
        return model
