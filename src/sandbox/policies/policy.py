
from abc import ABC, abstractmethod
from typing import Iterable

from sandbox.enviroments.base.observation import Observation

from sandbox.value_estimates.value_estimates import Action, ValueEstimates


class Policy(ABC):
    """
    A decision-making rule.

    Given current observation, possible actions, and algorithms memories (values esitmates) makes a decision about next step.
    """
    @abstractmethod
    def __call__(self, observation: Observation, actions: Iterable[Action], value_estimates: ValueEstimates):
        ...


class DeterministicPolicy(Policy):
    @abstractmethod
    def __call__(self, observation: Observation, actions: Iterable[Action], value_estimates: ValueEstimates) -> Action:
        """
        Returns one action of :param actions:
        """
        return super().__call__(observation, actions, value_estimates)


class StochasticPolicy(Policy):
    @abstractmethod
    def __call__(self, observation: Observation, actions: Iterable[Action], value_estimates: ValueEstimates) -> list[float]:
        """
        Returns probabilities of action selection for every action from :param actions:
        """
        return super().__call__(observation, actions, value_estimates)
