from __future__ import annotations
from enum import Enum
from typing import Sequence, Tuple, Set


class GridMove(Enum):
    __order__ = 'N E S W'
    N = (-1, 0)
    E = (0, 1)
    S = (1, 0)
    W = (0, -1)
    

    def __str__(self) -> str:
        return self.name
    
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return self.value[index]

