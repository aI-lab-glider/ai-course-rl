from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from random import random
from typing import Callable
import gym
from matplotlib import pyplot as plt
from sandbox.algorithms.algorithm import Algorithm
from sandbox.policies.policy import Policy
from sandbox.wrappers.named_env_wrapper import NamedEnv
from sandbox.wrappers.stats_wrapper import PlotType, StatsWrapper
import distinctipy


@dataclass
class EnvPolicies:
    env: NamedEnv
    policies: list[Policy] = field(default_factory=list)


class Comparator:
    def compare_algorithms(self, algorithms: list[Algorithm], envs: list[NamedEnv], get_algorithm_label: Callable[[Algorithm], str], n_episodes: int = 5000, plot_types: list[PlotType] = None) -> list[EnvPolicies]:
        plot_types = plot_types or list(PlotType)
        _, axs = plt.subplots(len(envs), len(
            plot_types), figsize=(10, 10), squeeze=False)
        algo_colors = distinctipy.get_colors(len(algorithms))
        policies_for_all_envs = []
        for i, env in enumerate(envs):
            env_axs = axs[i]
            training_results = EnvPolicies(env)
            policies_for_all_envs.append(training_results)
            for algo, color in zip(algorithms, algo_colors):
                env = deepcopy(env)
                env = StatsWrapper(env)
                policy = algo.run(n_episodes, env)
                env.plot(types=plot_types, ax=env_axs, color=color)
                training_results.policies.append(policy)
            for ax in env_axs:
                ax.legend([get_algorithm_label(a) for a in algorithms])

        # plt.show()
        return policies_for_all_envs

    def compare_policies(self, envs_with_policies: list[EnvPolicies], n_episodes: int, max_episode_length=10000):
        for ep in envs_with_policies:
            original_env = ep.env
            for policy in ep.policies:
                env = StatsWrapper(original_env, False)
                for _ in range(n_episodes):
                    observation = env.reset()
                    is_done = False
                    while not is_done and env.steps_count < max_episode_length:
                        action = policy.select_action(observation)
                        observation, _, is_done, _ = env.step(action)
                policy_summary = f"""
                    Environment: {env.name}
                    Policy: {repr(policy)}
                    N episodes: {n_episodes}
                    Average steps per episode: {env.average_step_count()}
                    Average reward per episode: {env.average_reward()}
                """
                logging.info(policy_summary)
