from typing import Callable

import numpy as np
from sandbox.algorithms.algorithm import Observation
from sandbox.value_estimates.value_estimates import Action, ActionValueEstimates, StateValueEstimates

class TabularStateValueEstimates(StateValueEstimates):    
    def __init__(self) -> None:
        self._model = {}
    
    def get(self, observations: list[Observation]) -> dict[Observation, float]:
        return {o: self._model[o] for o in observations}




    

class TabularActionValueEstimates(ActionValueEstimates):
    def __init__(self) -> None:
        self._model = {}

    def get(self, observation: Observation, actions: list[Action]) -> dict[Action, float]:
        return {a: self._model[observation, a] for a in actions}


