from typing import Optional, Union, Tuple
import gym
from gym.core import ObsType, ActType
import numpy as np
import random

from sandbox.enviroments.twenty_forty_eight.game.game import TwentyFortyEightGame
from sandbox.enviroments.twenty_forty_eight.game.heuristic import TwentyFortyEightHeuristic
from sandbox.enviroments.twenty_forty_eight.game.action import TwentyFortyEightPlayerAction, Direction


class TwentyFortyEightEnv(gym.Env):
    def __init__(self):
        self.game = TwentyFortyEightGame()
        self.heuristic = TwentyFortyEightHeuristic(self.game)
        self.observation_space = gym.spaces.Box(0, 2048, (self.game.board_dim, ), dtype=np.uint32)
        self.action_space = gym.spaces.Discrete(4)
        self.state = self.game.initial_state

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        new_state = self.game.take_action(self.state, self.get_action(action))
        new_state = self.game.take_action(new_state, random.choice(self.game.actions_for(new_state, is_opponent=True)))
        done = self.game.is_terminal_state(new_state)
        reward = self.heuristic(new_state) - self.heuristic(self.state)
        self.state = new_state
        info = {}
        return self.state.flatten(), reward, done, info

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[
        ObsType, tuple[ObsType, dict]]:
        self.state = self.game.initial_state
        return self.state.flatten()

    def render(self, mode="human"):
        pass

    @staticmethod
    def get_action(action: ActType) -> TwentyFortyEightPlayerAction:
        directions = {0: Direction.LEFT, 1: Direction.RIGHT, 2: Direction.UP, 3: Direction.DOWN}
        return TwentyFortyEightPlayerAction(direction=directions[action])
