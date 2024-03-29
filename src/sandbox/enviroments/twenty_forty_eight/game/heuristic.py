import numpy as np

from sandbox.enviroments.twenty_forty_eight.game.base.heuristic import Heuristic
from sandbox.enviroments.twenty_forty_eight.game.state import TwentyFortyEightState
from sandbox.enviroments.twenty_forty_eight.game.game import TwentyFortyEightGame

MAX_TILE_CREDIT = 2048
WEIGHT_MATRIX = np.array([
    [2048, 1024, 64, 32],
    [512, 128, 16, 2],
    [256, 8, 2, 1],
    [4, 2, 1, 1]
])


class TwentyFortyEightHeuristic(Heuristic[TwentyFortyEightState]):
    def __init__(self, game: TwentyFortyEightGame) -> None:
        self.game = game

    def __call__(self, state: TwentyFortyEightState) -> float:
        if self.game.is_terminal_state(state):
            return np.inf * self.game.reward(state)
        return self.empty_tiles(state) + self.max_tile_position(state) + self.weighted_board(state) \
               - self.smoothness(state) + self._monotonicity(state)

    def empty_tiles(self, state: TwentyFortyEightState) -> int:
        return 512 * np.count_nonzero(state.board == 0)

    def max_tile_position(self, state: TwentyFortyEightState) -> int:
        max_tile = np.amax(state.board)
        if state.board[0][0] == max_tile:
            return MAX_TILE_CREDIT
        else:
            return -MAX_TILE_CREDIT

    def weighted_board(self, state: TwentyFortyEightState) -> int:
        return (WEIGHT_MATRIX * state.board).sum()

    def smoothness(self, state: TwentyFortyEightState) -> int:
        smoothness = sum(abs(state.board[row][col] - state.board[row + 1][col])
                         for row in range(state.shape[0] - 1) for col in range(state.shape[1]))
        smoothness += sum(abs(state.board[row][col] - state.board[row][col + 1])
                          for row in range(state.shape[0]) for col in range(state.shape[1] - 1))
        return 1024 * smoothness

    def _monotonicity(self, state: TwentyFortyEightState) -> int:
        monotonicity = sum(state.board[row + 1][col]
                           for col in range(state.shape[1]) for row in range(state.shape[0] - 1)
                           if state.board[row][col] > state.board[row + 1][col])
        monotonicity += sum(state.board[row][col + 1]
                            for col in range(state.shape[1] - 1) for row in range(state.shape[0])
                            if state.board[row][col] > state.board[row][col + 1])
        return 128 * monotonicity
