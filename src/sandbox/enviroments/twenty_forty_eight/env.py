from typing import Optional, Union, Tuple
from numpy.typing import NDArray
import gym
import numpy as np
import random
from math import log
from gym.core import ObsType, ActType

from sandbox.enviroments.twenty_forty_eight.game.game import TwentyFortyEightGame
from sandbox.enviroments.twenty_forty_eight.game.state import TwentyFortyEightState
from sandbox.enviroments.twenty_forty_eight.game.heuristic import TwentyFortyEightHeuristic
from sandbox.enviroments.twenty_forty_eight.game.action import TwentyFortyEightPlayerAction, Direction


class TwentyFortyEightEnv(gym.Env):
    def __init__(self, invalid_move_threshold=16, invalid_move_percentage=0.1):
        self.game = TwentyFortyEightGame()
        self.heuristic = TwentyFortyEightHeuristic(self.game)
        self.state = self.game.initial_game_state()
        self.observation_space = gym.spaces.MultiDiscrete([12 for _ in range(self.game.board_dim**2)], dtype=np.uint32)
        self.action_space = gym.spaces.Discrete(4)
        self.invalid_move_threshold = invalid_move_threshold
        self.invalid_move_percentage = invalid_move_percentage
        self._total_action_count = 0
        self._invalid_action_count = 0
        self._images = []

    def step(self, action: int) -> Tuple[str, float, bool, dict]:
        new_state = self._update_state(action)
        reward = self.heuristic(new_state) - self.heuristic(self.state)
        done = self.game.is_terminal_state(new_state)
        if self._too_many_invalid():
            done = True
            reward = -np.inf
        self.state = new_state
        observation = self.to_exponents(self.state.board.flatten())
        info = {}
        return str(observation), float(reward), done, info

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[
        NDArray, tuple[NDArray, dict]]:
        self.state = self.game.initial_game_state()
        self._total_action_count = 0
        self._invalid_action_count = 0
        self._images = [self.game.to_image(self.state)]
        return self.to_exponents(self.state.board.flatten())

    def render(self, mode="human"):
        self._images.append(self.game.to_image(self.state))

    def _update_state(self, action: int) -> TwentyFortyEightState:
        self._total_action_count += 1
        game_action = self.get_action(action)
        if self.game._is_valid_move(game_action.direction, self.state):
            new_state = self.game.take_action(self.state, game_action)
            new_state = self.game.take_action(new_state,
                                              random.choice(self.game.actions_for(new_state, is_opponent=True)))
        else:
            self._invalid_action_count += 1
            new_state = self.state
        return new_state

    def _too_many_invalid(self) -> bool:
        return self._invalid_action_count > self.invalid_move_threshold \
            and self._invalid_action_count > self._total_action_count * self.invalid_move_percentage

    def to_gif(self, img_name="2048_env_game"):
        # print("AAAAAAAAAAAAAAAAAAAA we half way there")
        self._images[0].save(img_name, save_all=True, append_images=self._images[1:],
                             format='GIF', optimize=False, duration=500, loop=1)
        print(self._images[0])


    @staticmethod
    def get_action(action: int) -> TwentyFortyEightPlayerAction:
        directions = {0: Direction.LEFT, 1: Direction.RIGHT, 2: Direction.UP, 3: Direction.DOWN}
        return TwentyFortyEightPlayerAction(direction=directions[action])

    @staticmethod
    def to_exponents(flatten_board: NDArray):
        return str(np.array([log(x, 2) if x != 0 else 0 for x in flatten_board], np.uint32))
