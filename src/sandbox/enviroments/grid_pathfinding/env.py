from random import choice, random
from typing import overload
import gym
from gym import spaces

import numpy as np
from math import sqrt

from sandbox.enviroments.grid_pathfinding.problem.grid import GridCoord
from sandbox.enviroments.grid_pathfinding.problem.gird_pathfinding import GridPathfinding, Grid, GridCoord, GridMove
import operator as op

class GridPathfindingEnv(gym.Env):
    @overload
    def __init__(self, file, goal_transition_prob: float=0, diagonal_weight: float = 0):
        ...

    @overload
    def __init__(self, grid: Grid, initial: GridCoord, goal: GridCoord, goal_transition_prob: float=0, diagonal_weight: float = 0):
        ...

    def __init__(self, file=None, grid: Grid=None, initial: GridCoord=None, goal: GridCoord=None, goal_transition_prob: float=0, diagonal_weight: float = 0) -> None:
        super().__init__()

        self.problem = GridPathfinding.from_file(file) if file is not None else GridPathfinding(grid, initial, goal, diagonal_weight)
        
        self.size = self.problem.grid.board.size
        self.action_space = gym.spaces.Discrete(len(GridMove))
        self.observation_space = spaces.Tuple(
            tuple({
                "position": spaces.Discrete(self.size),
                "target": spaces.Discrete(self.size),
            }.values())
        )

        self._agent_location = initial
        self._goal_transition_prob = goal_transition_prob



    def step(self, action: int) -> tuple[dict[str, GridCoord], float, bool, dict[str, float]]:
        direction = list(GridMove)[action]

        if self._is_legal_move(self._agent_location, direction):
            self._agent_location = self._agent_location + direction
            done = self.problem.is_goal(self._agent_location)
        else:
            done = False
        
        if random() < self._goal_transition_prob:
            goal = self.problem.goal
            valid_goal_transitions = [m for m in GridMove if self._is_legal_move(goal, m)]
            self.problem.goal  = self._take_action(goal, choice(valid_goal_transitions))

        reward = 1 if done else 0
        info = self._get_info()
        return self._get_obs(), float(reward), done, info
    
    def _is_legal_move(self, location: GridCoord, move: GridMove):
        return self.problem.is_legal_move(location, move)
    
    def _take_action(self, location: GridCoord, move: GridMove):
        return self.problem.take_action(location, move)
    
    def _flatten_location(self, location: GridCoord):
        return location.x + location.y * self.problem.grid.shape[0]
    

    def reset(self, seed=None, return_info=False, options=None) -> dict[str, float] or dict[str, GridCoord]:
        self._agent_location = self.problem.initial

        observation = self._get_obs()
        info = self._get_info()

        return (observation, info) if return_info else observation

    def render(self, mode):
        if mode == "ansi":
            return self.problem.to_str(self._agent_location)


    def _get_obs(self) -> dict[str, GridCoord]:
        return tuple({
            "position": self._flatten_location(self._agent_location),
            "target": self._flatten_location(self.problem.goal)
        }.values())

    def _get_info(self) -> dict[str, float]:
        return {"distance": self.calculate_distance(self._agent_location)}

    def calculate_distance(self, state: GridCoord) -> float:
        target = self.problem.goal
        return sqrt((state.x - target.x) ** 2 + (state.y - target.y) ** 2)
