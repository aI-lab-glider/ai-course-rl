import logging
import sys
from pathlib import Path
import numpy as np
import gym
path = Path(__file__)
sys.path.append(str(path.parents[1].absolute()))

from sandbox.action_selection_rules.epsilon_greedy import EpsilonGreedyActionSelection
from sandbox.algorithms.dqn import DQNAlgorithm, MyQNetwork


def main():
    logging.basicConfig(level=logging.INFO)
    env = gym.make("CartPole-v1")
    dqn = DQNAlgorithm(
        memory_size=2000,
        create_network=MyQNetwork,
        action_selection_rule=EpsilonGreedyActionSelection(0.01),
        learning_rate=1e-2,
        max_episode_length=200,
        exploration_steps=50,
        target_update_steps=50,
        batch_size=32,
        discount_rate=0.95,
    )
    q_network = dqn.run(
        n_episodes=60,
        env=env
    )

    enjoy(
        env=env,
        action_selection=lambda state: int(np.argmax(q_network.predict(state))),
        steps=2_000
    )

def enjoy(env, action_selection, steps) -> None:
    state = env.reset()
    for step in range(steps):
        action = action_selection(state)
        state, reward, done, info = env.step(action)
        if done:
            state = env.reset()
        img = env.render()



if __name__ == "__main__":
    main()
