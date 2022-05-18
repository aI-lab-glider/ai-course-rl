import gym
from gym import spaces

import numpy as np
from problem.gird_pathfinding import GridPathfinding, Grid, GridCoord, GridMove
from math import sqrt

from sandbox.enviroments.grid_pathfinding.problem.grid import GridCoord


class GridPathfindingEnv(gym.Env):
    def __init__(self, grid: Grid, initial: GridCoord, goal: GridCoord, diagonal_weight: float = 0):

        self.problem = GridPathfinding(grid, initial, goal, diagonal_weight)
        self.size = self.problem.grid.board.size

        self.action_space = gym.spaces.Discrete(8)

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0, 0]), high=np.array([self.size[0], self.size[1]]),
                                    dtype=np.float32),
                "target": spaces.Box(low=np.array([0, 0]), high=np.array([self.size[0], self.size[1]]),
                                     dtype=np.float32)
            }
        )

        self._action_to_direction = {idx: np.array(list(coord.value)) for (idx, coord) in enumerate(GridMove)}
        self._agent_location = initial
        self._target_location = goal

    def step(self, action: int) -> tuple[dict[str, GridCoord], float, bool, dict[str, float]]:
        direction = self._action_to_direction[action]

        if self.problem.is_legal_move(self._agent_location, direction):
            self._agent_location = self._agent_location + direction
            done = self.problem.is_goal(self._agent_location)
        else:
            done = False

        reward = 1 if done else 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, float(reward), done, info

    def reset(self, seed=None, return_info=False, options=None) -> dict[str, float] or dict[str, GridCoord]:
        self._agent_location = self.problem.initial

        observation = self._get_obs()
        info = self._get_info()

        return (observation, info) if return_info else observation

    def render(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def _get_obs(self) -> dict[str, GridCoord]:
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self) -> dict[str, float]:
        return {"distance": self.calculate_distance(self._agent_location)}

    def calculate_distance(self, state: GridCoord) -> float:
        return sqrt((state.x - self._target_location.x) ** 2 + (state.y - self._target_location.y) ** 2)
