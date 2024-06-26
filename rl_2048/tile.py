from collections import defaultdict
from copy import deepcopy
from random import SystemRandom
from typing import NamedTuple

from rl_2048.common import Location


class MovingGrid(NamedTuple):
    src_loc: Location
    src_val: int
    dst_loc: Location
    dst_val: int


class Tile:
    def __init__(self, width: int = 4, height: int = 4):
        self.width: int = width
        self.height: int = height
        self.grids: list[list[int]] = []
        self.animation_grids: dict[Location, list[MovingGrid]] = defaultdict(list)
        self._cryptogen: SystemRandom = SystemRandom()
        self.random_start_count: int = 2

        self.random_start()

    def flattened(self) -> list[int]:
        return [g for row in self.grids for g in row]

    def __repr__(self) -> str:
        s: str = "-" * (8 * self.width + 1) + "\n"
        for row in self.grids:
            s += "|"
            for g in row:
                s += f" {g:05d} |"
            s += "\n" + "-" * (8 * self.width + 1) + "\n"
        return s

    def set_grids(self, grids: list[list[int]]):
        if len(grids) != self.height:
            raise ValueError(
                f"Wrong grids height, expected {self.height}, actual {len(grids)}"
            )
        for row in grids:
            if len(row) != self.width:
                raise ValueError(
                    f"Wrong grids width, expected {self.width}, actual {len(row)}"
                )

        self.grids = deepcopy(grids)

    def random_start(self):
        self.grids = [[0 for _x in range(self.width)] for _y in range(self.height)]
        self.reset_animation_grids()

        def fill_grid(index):
            loc = Location(x=index % self.width, y=index // self.width)
            val = 2 ** self._cryptogen.randint(1, 2)
            self.grids[loc.y][loc.x] = val
            self.animation_grids[loc].append(MovingGrid(loc, 0, loc, val))

        random_indices = self._cryptogen.sample(
            range(self.width * self.height), self.random_start_count
        )
        for index in random_indices:
            fill_grid(index)

    def max_grid(self) -> int:
        return max(r for rows in self.grids for r in rows)

    def reset_animation_grids(self):
        self.animation_grids = defaultdict(list)
