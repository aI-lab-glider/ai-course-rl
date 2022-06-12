from __future__ import annotations
from enum import Enum
from typing import Sequence, Tuple, Set


class GridMove(Enum):
    __order__ = 'N S W E NW NE SW SE'
    N = (-1, 0)
    S = (1, 0)
    W = (0, -1)
    E = (0, 1)
    NW = (-1, -1)
    NE = (-1, 1)
    SW = (1, -1)
    SE = (1, 1)

    @staticmethod
    def diagonal_moves() -> Set[GridMove]:
        return {GridMove.NW, GridMove.NE, GridMove.SW, GridMove.SE}

    def involved_moves(self) -> Sequence[GridMove]:
        if self not in GridMove.diagonal_moves():
            return [self]
        shift_l = (0, self.value[1])
        shift_r = (self.value[0], 0)
        return [self, GridMove(shift_l), GridMove(shift_r)]

    def __str__(self) -> str:
        return self.name
    
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return self.value[index]

