from itertools import count
import gym, random
import gym.spaces
import numpy as np
from collections import namedtuple, deque
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential


import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)
import matplotlib.animation as animation

mpl.rc("animation", html="jshtml")


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return (patch,)


def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis("off")
    anim = animation.FuncAnimation(
        fig,
        update_scene,
        fargs=(frames, patch),
        frames=len(frames),
        repeat=repeat,
        interval=interval,
    )
    plt.show()


def build_network(input_shape: list[int], output_shape: int) -> Sequential:
    model = Sequential(
        [
            keras.layers.Dense(32, activation="elu", input_shape=[4]),
            keras.layers.Dense(32, activation="elu"),
            keras.layers.Dense(output_shape),
        ]
    )
    return model


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAlgorithm:
    def __init__(self, env: gym.Env, replay_memory_len: int):
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), f"DQN algorithm supports only discrete action space"
        self.env = env
        self.replay_buffer = ReplayMemory(replay_memory_len)
        self.network = build_network(
            self.env.observation_space.high.shape, self.env.action_space.n
        )
        self.target_network = keras.models.clone_model(self.network)
        self.loss_fn = keras.losses.mean_squared_error

    def train(
        self,
        n_episodes: int,
        max_episode_length: int,
        exploration_steps: int,
        target_update_steps: int,
        batch_size: int = 32,
        discount_rate: float = 0.95,
        learning_rate: float = 1e-2,
    ):
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.discount_rate = discount_rate
        self.episodes_length = []

        for episode in range(n_episodes):
            sequence = []
            obs = self.env.reset()
            for t in range(max_episode_length):
                epsilon = max(1 - episode / 500, 0.01)
                obs, reward, done, info = self._play_one_step(obs, epsilon)
                if done:
                    break
            self.episodes_length.append(t)
            if len(self.replay_buffer) <= exploration_steps:
                continue

            self._training_step(batch_size)
            if episode % target_update_steps == 0:
                self.target_network.set_weights(self.network.get_weights())
        self._plot()

    def enjoy(self) -> None:
        self.env.seed(43)
        state = self.env.reset()

        frames = []

        for step in range(200):
            action = self._epsilon_greedy_policy(state, 0)
            state, reward, done, info = self.env.step(action)
            if done:
                break
            img = self.env.render(mode="rgb_array")
            frames.append(img)

        plot_animation(frames)

    def _epsilon_greedy_policy(self, state, epsilon) -> int:
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        Q_values = self.network.predict(state[np.newaxis])
        return int(np.argmax(Q_values))

    def _plot(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.episodes_length)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Sum of rewards", fontsize=14)
        plt.show()

    def _play_one_step(
        self, state, epsilon: float
    ) -> tuple[np.ndarray, float, bool, dict]:
        action = self._epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, info = self.env.step(action)
        self.replay_buffer.push(state, action, next_state, reward, done)
        return next_state, reward, done, info

    def _training_step(self, batch_size: int) -> None:
        transitions = self.replay_buffer.sample(batch_size)
        states, actions, next_states, rewards, dones = list(
            map(np.array, zip(*transitions))
        )


        next_Q_values = self.network.predict(next_states)
        best_next_actions = np.argmax(next_Q_values, axis=1)

        next_mask = tf.one_hot(best_next_actions, self.env.action_space.n).numpy()
        next_best_Q_values = (self.target_network.predict(next_states) * next_mask).sum(
            axis=1
        )

        target_Q_values = (
            rewards + (1 - dones) * self.discount_rate * next_best_Q_values
        )
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.env.action_space.n)
        with tf.GradientTape() as tape:
            all_Q_values = self.network(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))


if __name__ == "__main__":
    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    env = gym.make("CartPole-v1")
    dqn = DQNAlgorithm(env, 2000)
    dqn.train(60, 200, 50, 50)
    dqn.enjoy()
