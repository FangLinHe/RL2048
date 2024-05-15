from collections import defaultdict
from random import randint, sample
from typing import Dict, List, NamedTuple

from RL2048.common import Location


class MovingGrid(NamedTuple):
    src_loc: Location
    src_val: int
    dst_loc: Location
    dst_val: int


class Tile:
    def __init__(self, width: int = 4, height: int = 4):
        self.width: int = width
        self.height: int = height
        self.grids: List[List[int]] = []
        self.animation_grids: Dict[Location, List[MovingGrid]] = defaultdict(list)
        self.random_start()

    def random_start(self):
        self.grids = [[0 for _x in range(self.width)] for _y in range(self.height)]
        self.reset_animation_grids()

        def fill_grid(index):
            loc = Location(x=index % self.width, y=index // self.width)
            val = 2 ** randint(1, 2)
            self.grids[loc.y][loc.x] = val
            self.animation_grids[loc].append(MovingGrid(loc, 0, loc, val))

        random_count = 2
        random_indices = sample(range(self.width * self.height), random_count)
        assert len(set(random_indices)) == random_count
        for index in random_indices:
            fill_grid(index)

    def max_grid(self) -> int:
        return max(r for rows in self.grids for r in rows)

    def reset_animation_grids(self):
        self.animation_grids = defaultdict(list)
