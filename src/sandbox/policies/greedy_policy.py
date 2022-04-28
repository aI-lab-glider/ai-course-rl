

from typing import Iterable
from sandbox.algorithms.algorithm import Action
from sandbox.enviroments.base.observation import Observation
from sandbox.enviroments.multi_armed_bandit.policy import DeterministicPolicy
from sandbox.value_estimates.value_estimates import ActionValueEstimates, StateValueEstimates

class GreedyPolicy(DeterministicPolicy):

    def _policy_for_action_value_estimate(self, observation: Observation, actions: Iterable[Action], estimates: ActionValueEstimates):
        return max([(action, estimates[(observation, action)])
            for action in actions
        ], key=lambda i: i[1])[0]

    def _policy_for_value_estimate(self, next_observations: Iterable[Observation], estimates: StateValueEstimates):
        return max([(observation, estimates[observation])
            for observation in next_observations
        ], key=lambda i: i[1])[0]


    
