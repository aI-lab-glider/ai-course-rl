from typing import Callable, Hashable, Iterable, MutableMapping

import numpy as np
from sandbox.algorithms.algorithm import Observation
from sandbox.value_estimates.value_estimates import Action, ActionValueEstimates, State, StateValueEstimates, ValueEstimates
from collections import UserDict
from gym.spaces.space import Space

class TabularStateValueEstimates(UserDict, StateValueEstimates):
    def __init__(self, observations: Iterable[Observation], initialization_method: Callable[[Observation], float] = lambda _: np.random.random()) -> None:
        super().__init__({
            s: initialization_method(s)
            for s in observations
        })


class TabularActionValueEstimates(UserDict, ActionValueEstimates):
    def __init__(self, observation_actions_map: dict[Observation, Iterable[Action]], initialization_method: Callable[[Observation, Action], float] = lambda *_: np.random.random()) -> None:
        super().__init__({
            (state, action): initialization_method(state, action)
            for state in observation_actions_map
            for action in observation_actions_map[state]
        })
