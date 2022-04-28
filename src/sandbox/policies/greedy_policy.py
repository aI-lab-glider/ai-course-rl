
from sandbox.enviroments.base.observation import Observation
from sandbox.policies.policy import ActionValuePolicy, StateValuePolicy
from sandbox.value_estimates.value_estimates import ActionValueEstimates, Action, StateValueEstimates


class GreedyActionValuePolicy(ActionValuePolicy[ActionValueEstimates]):
    def __init__(self) -> None:
        super().__init__(ActionValueEstimates())

    def get(self, observation: Observation, actions: list[Action]) -> Action:
        estimates = self._action_value_estimates.get(observation, actions)
        return max(estimates, key=lambda action: estimates[action])


class GreedyStateValuePolicy(StateValuePolicy[StateValueEstimates]):
    def __init__(self) -> None:
        super().__init__(StateValueEstimates())


    def get(self, observation: list[Observation]) -> Observation:
        estimates = self._value_estimates.get(observation, observation)
        return max(estimates, key=lambda observation: estimates[observation])

    
