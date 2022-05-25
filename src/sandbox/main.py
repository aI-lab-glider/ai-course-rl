import sys
from pathlib import Path
import numpy as np
import gym

path = Path(__file__)
sys.path.append(str(path.parents[1].absolute()))

from sandbox.enviroments.multi_armed_bandit import (
    BanditEnv,
    BanditTrainer,
    EpsilonGreedy,
    UCB,
    ThompsonSampling,
)
from sandbox.enviroments.multi_armed_bandit.env import NormalDistribution
from sandbox.algorithms.dqn import DQNAlgorithm, MyQNetwork, epsilon_greedy_policy


def main():
    env = gym.make("CartPole-v1")
    dqn = DQNAlgorithm(
        env,
        memory_size=2000,
        network=MyQNetwork,
        policy=lambda *args: epsilon_greedy_policy(0.01, *args),
        learning_rate=1e-2,
    )
    q_network = dqn.train(
        n_episodes=60,
        max_episode_length=200,
        exploration_steps=50,
        target_update_steps=50,
        batch_size=32,
        discount_rate=0.95,
    )

    enjoy(
        env=env,
        policy=lambda state: int(np.argmax(q_network.predict(state))),
        steps=2_000
    )

def enjoy(env, policy, steps) -> None:
    state = env.reset()
    for step in range(steps):
        action = policy(state)
        state, reward, done, info = env.step(action)
        if done:
            state = env.reset()
        img = env.render()



if __name__ == "__main__":
    main()
