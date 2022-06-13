from __future__ import annotations
import stat
from typing_extensions import Self
from sandbox.enviroments.base.problem import Problem
from sandbox.enviroments.base.problem import ReversibleProblem
from sandbox.enviroments.grid_pathfinding.problem.grid import Grid, GridCell, GridCoord
from sandbox.enviroments.grid_pathfinding.problem.grid_move import GridMove
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw



class GridPathfinding(ReversibleProblem[GridCoord, GridMove]):
    def __init__(self, grid: Grid, initial: GridCoord, goal: GridCoord):
        super().__init__(initial, goal)
        self.grid = grid

    def actions(self, state: GridCoord) -> List[GridMove]:
        return [a for a in GridMove if self.is_legal_move(state, a)]

    def is_legal_move(self, coord: GridCoord, move: GridMove) -> bool:
        new_coord = coord + move.value
        if not (0 <= new_coord.x < self.grid.shape[1]):
            return False
        if not (0 <= new_coord.y < self.grid.shape[0]):
            return False

        new_location = self.grid.get_cell(new_coord)
        if new_location == GridCell.WALL:
            return False

        return True

    def take_action(self, state: GridCoord, action: GridMove) -> GridCoord:
        return state + action.value

    def action_cost(self, source: GridCoord, action: GridMove) -> float:
        return 1.0

    def is_goal(self, state: GridCoord):
        return state == self.goal

    def reversed(self):
        return GridPathfinding(self.grid, self.goal, self.initial)

    def to_image(self, state: GridCoord, size: Tuple[int, int] = (800, 800)) -> Image.Image:
        pass
        # image = Image.new("RGB", size, (248, 255, 229))
        # grid_drawer = GridDrawer(image, self.grid)
        # grid_drawer.draw_grid()
        # for y, row in enumerate(self.grid):
        #     for x, cell in enumerate(row):
        #         if cell == GridCell.WALL:
        #             grid_drawer.draw_rectangle((x, y), fill=(
        #                 31, 122, 140), padding=-grid_drawer.border)
        # grid_drawer.draw_circle(self.goal.x, self.goal.y, fill=(255, 100, 100))
        # grid_drawer.draw_circle(state.x, state.y, (100, 100, 100))
        # return image
    
    def to_str(self, agent_location: GridCoord) -> str:
        grid = ""
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if agent_location == GridCoord(x, y):
                    grid += " A "
                elif self.goal == GridCoord(x, y):
                    grid += " G "
                elif cell == GridCell.WALL:
                    grid += " W "
                else: 
                    grid += " . "
            grid += "\n"
        return grid


    @classmethod
    def from_file(cls, file) -> Self:
        with open(file) as f:
            lines = f.read()
            return cls.deserialize(lines)

    @staticmethod
    def deserialize(text: str) -> Self:
        lines = text.splitlines()
        header = lines[0]
        raw_width = header.strip()
        width = int(raw_width)
        raw_grid = [l[1:] for l in lines[1:] if l.startswith("|")]

        start: Optional[GridCoord] = None
        goal: Optional[GridCoord] = None
        board = np.full((len(raw_grid), width), GridCell.EMPTY)

        for y, row in enumerate(raw_grid):
            for x, cell in enumerate(row):
                if cell.upper() == "S":
                    start = GridCoord(x, y)
                elif cell.upper() == "G":
                    goal = GridCoord(x, y)
                elif cell == GridCell.WALL.value:
                    board[y, x] = GridCell.WALL

        assert start is not None, "grid is missing a start cell 'S'"
        assert goal is not None, "grid is missing a goal cell 'G'"
        return GridPathfinding(Grid(board), start, goal)

