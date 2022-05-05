from typing import Optional, Union, Tuple
from numpy.typing import NDArray
import gym
import numpy as np
import random

from sandbox.enviroments.twenty_forty_eight.game.game import TwentyFortyEightGame
from sandbox.enviroments.twenty_forty_eight.game.state import TwentyFortyEightState
from sandbox.enviroments.twenty_forty_eight.game.heuristic import TwentyFortyEightHeuristic
from sandbox.enviroments.twenty_forty_eight.game.action import TwentyFortyEightPlayerAction, Direction


class TwentyFortyEightEnv(gym.Env):
    def __init__(self, invalid_move_threshold=16, invalid_move_percentage=0.1):
        self.game = TwentyFortyEightGame()
        self.heuristic = TwentyFortyEightHeuristic(self.game)
        self.state = self.game.initial_game_state()
        self.observation_space = gym.spaces.Box(0, 2048, (self.game.board_dim*self.game.board_dim, ), dtype=np.uint32)
        self.action_space = gym.spaces.Discrete(4)
        self.invalid_move_threshold = invalid_move_threshold
        self.invalid_move_percentage = invalid_move_percentage
        self._total_count = 0
        self._invalid_count = 0

    def step(self, action: int) -> Tuple[NDArray, float, bool, dict]:
        new_state = self._update_state(action)
        reward = self.heuristic(new_state) - self.heuristic(self.state)
        done = self.game.is_terminal_state(new_state)
        if self._too_many_invalid():
            done = True
            reward = -np.inf
        self.state = new_state
        observation = self.state.board.flatten().astype(np.uint32)
        info = {}
        return observation, float(reward), done, info

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[
        NDArray, tuple[NDArray, dict]]:
        self.state = self.game.initial_game_state()
        self._total_count = 0
        self._invalid_count = 0
        return self.state.board.flatten().astype(np.uint32)

    def render(self, mode="human"):
        pass

    def _update_state(self, action: int) -> TwentyFortyEightState:
        self._total_count += 1
        game_action = self.get_action(action)
        if self.game._is_valid_move(game_action.direction, self.state):
            new_state = self.game.take_action(self.state, game_action)
            new_state = self.game.take_action(new_state,
                                              random.choice(self.game.actions_for(new_state, is_opponent=True)))
        else:
            self._invalid_count += 1
            new_state = self.state
        return new_state

    def _too_many_invalid(self) -> bool:
        return self._invalid_count > self.invalid_move_threshold \
            and self._invalid_count > self._total_count * self.invalid_move_percentage

    @staticmethod
    def get_action(action: int) -> TwentyFortyEightPlayerAction:
        directions = {0: Direction.LEFT, 1: Direction.RIGHT, 2: Direction.UP, 3: Direction.DOWN}
        return TwentyFortyEightPlayerAction(direction=directions[action])
