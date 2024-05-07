from collections import namedtuple
from random import randint, sample
from RL2048.tile import Tile

from typing import List, Tuple, NamedTuple


class Location(NamedTuple):
    x: int
    y: int


class MoveResult(NamedTuple):
    suc: bool
    score: int


class GameEngine:
    def __init__(self, tile: Tile):
        self.tile = tile
        self.score = 0

    def move_up(self) -> MoveResult:
        suc: bool = False
        score: int = 0

        # merge grids up if values are the same as the grid above
        # and values are added up
        for x in range(self.tile.width):
            above = -1
            y = 0
            while y < self.tile.height:
                if self.tile.grids[y][x] > 0:
                    if (
                        above == -1
                        or self.tile.grids[y][x] != self.tile.grids[above][x]
                    ):
                        above = y
                    else:
                        self.tile.grids[above][x] *= 2
                        self.tile.grids[y][x] = 0
                        above = -1

                        suc = True
                        score += self.tile.grids[above][x]
                y += 1

        # move all the grids up
        for x in range(self.tile.width):
            next_y = 0
            for y in range(self.tile.height):
                if self.tile.grids[y][x] == 0:
                    continue

                self.tile.grids[next_y][x] = self.tile.grids[y][x]

                if next_y != y:
                    self.tile.grids[y][x] = 0
                    suc = True
                next_y += 1

        self.score += score
        return MoveResult(suc, score)

    def move_down(self) -> MoveResult:
        suc: bool = False
        score: int = 0

        # merge grids up if values are the same as the grid below
        # and values are added up
        last_row = self.tile.height - 1
        for x in range(self.tile.width):
            below = -1
            y = last_row
            while y >= 0:
                if self.tile.grids[y][x] > 0:
                    if (
                        below == -1
                        or self.tile.grids[y][x] != self.tile.grids[below][x]
                    ):
                        below = y
                    else:
                        self.tile.grids[below][x] *= 2
                        score += self.tile.grids[below][x]
                        self.tile.grids[y][x] = 0
                        below = -1

                        suc = True
                        score += self.tile.grids[below][x]
                y -= 1

        # move all the grids down
        for x in range(self.tile.width):
            next_y = last_row
            for y in range(last_row, -1, -1):
                if self.tile.grids[y][x] == 0:
                    continue

                self.tile.grids[next_y][x] = self.tile.grids[y][x]

                if next_y != y:
                    self.tile.grids[y][x] = 0
                    suc = True
                next_y -= 1

        self.score += score
        return MoveResult(suc, score)

    def move_left(self) -> MoveResult:
        suc: bool = False
        score: int = 0

        # merge grids up if values are the same as the left grid
        # and values are added up
        for y in range(self.tile.height):
            left = -1
            x = 0
            while x < self.tile.width:
                if self.tile.grids[y][x] > 0:
                    if left == -1 or self.tile.grids[y][x] != self.tile.grids[y][left]:
                        left = x
                    else:
                        self.tile.grids[y][left] *= 2
                        score += self.tile.grids[y][left]
                        self.tile.grids[y][x] = 0
                        left = -1

                        suc = True
                        score += self.tile.grids[y][left]
                x += 1

        # move all the grids left
        for y in range(self.tile.height):
            next_x = 0
            for x in range(self.tile.width):
                if self.tile.grids[y][x] == 0:
                    continue

                self.tile.grids[y][next_x] = self.tile.grids[y][x]

                if next_x != x:
                    self.tile.grids[y][x] = 0
                    suc = True
                next_x += 1

        self.score += score
        return MoveResult(suc, score)

    def move_right(self) -> MoveResult:
        suc: bool = False
        score: int = 0

        # merge grids up if values are the same as the right grid
        # and values are added up
        last_col = self.tile.width - 1
        for y in range(self.tile.height):
            right = -1
            x = last_col
            while x >= 0:
                if self.tile.grids[y][x] > 0:
                    if (
                        right == -1
                        or self.tile.grids[y][x] != self.tile.grids[y][right]
                    ):
                        right = x
                    else:
                        self.tile.grids[y][right] *= 2
                        score += self.tile.grids[y][right]
                        self.tile.grids[y][x] = 0
                        right = -1

                        suc = True
                        score += self.tile.grids[y][right]
                x -= 1

        # move all the grids right
        for y in range(self.tile.height):
            next_x = last_col
            for x in range(last_col, -1, -1):
                if self.tile.grids[y][x] == 0:
                    continue

                self.tile.grids[y][next_x] = self.tile.grids[y][x]

                if next_x != x:
                    self.tile.grids[y][x] = 0
                    suc = True
                next_x -= 1

        self.score += score
        return MoveResult(suc, score)

    def find_empty_grids(self) -> List[Location]:
        empty_grids: List[Location] = []
        for y, grid_row in enumerate(self.tile.grids):
            for x, grid in enumerate(grid_row):
                if grid == 0:
                    empty_grids.append(Location(x, y))
            
        return empty_grids

    def generate_new(self) -> bool:
        empty_grids: List[Location] = self.find_empty_grids()
        if len(empty_grids) == 0:
            return False

        des: Location = sample(empty_grids, 1)[0]
        self.tile.grids[des.y][des.x] = 2 ** randint(1, 2)

        return True