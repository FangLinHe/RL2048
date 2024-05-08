from collections import namedtuple
from random import randint, sample
from RL2048.common import Location
from RL2048.tile import MovingGrid, Tile

from typing import List, NamedTuple


class MoveResult(NamedTuple):
    suc: bool
    score: int


class GameEngine:
    def __init__(self, tile: Tile):
        self.tile = tile
        self.score = 0
        
    def reset(self):
        self.tile.random_start()
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
                    # Candidate to be merged
                    if (
                        above == -1
                        or self.tile.grids[y][x] != self.tile.grids[above][x]
                    ):
                        above = y
                    # Merged with the grid above
                    else:
                        src_loc = Location(x, y)
                        dst_loc = Location(x, above)
                        val = self.tile.grids[above][x]
                        self.tile.animation_grids[dst_loc].append(MovingGrid(dst_loc, val, dst_loc, 0)) # Disappear after merging
                        self.tile.animation_grids[dst_loc].append(MovingGrid(src_loc, val, dst_loc, val*2)) # Twice after merging

                        self.tile.grids[above][x] *= 2
                        score += self.tile.grids[above][x]
                        self.tile.grids[y][x] = 0
                        above = -1

                        suc = True
                y += 1

        # move all the grids up
        for x in range(self.tile.width):
            next_y = 0
            for y in range(self.tile.height):
                if self.tile.grids[y][x] == 0:
                    continue

                if next_y != y:
                    src_loc = Location(x, y)
                    dst_loc = Location(x, next_y)
                    val = self.tile.grids[y][x]
                    if src_loc in self.tile.animation_grids:
                        # update destination location
                        assert dst_loc not in self.tile.animation_grids
                        self.tile.animation_grids[dst_loc] = [
                            MovingGrid(grid.src_loc, grid.src_val, dst_loc, grid.dst_val) \
                                for grid in self.tile.animation_grids[src_loc] 
                        ]
                        del self.tile.animation_grids[src_loc]
                    else:
                        self.tile.animation_grids[dst_loc].append(MovingGrid(src_loc, val, dst_loc, val))

                    self.tile.grids[next_y][x] = self.tile.grids[y][x]
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
                    # Candidate to be merged
                    if (
                        below == -1
                        or self.tile.grids[y][x] != self.tile.grids[below][x]
                    ):
                        below = y
                    # Merged with the grid above
                    else:
                        src_loc = Location(x, y)
                        dst_loc = Location(x, below)
                        val = self.tile.grids[below][x]
                        self.tile.animation_grids[dst_loc].append(MovingGrid(dst_loc, val, dst_loc, 0)) # Disappear after merging
                        self.tile.animation_grids[dst_loc].append(MovingGrid(src_loc, val, dst_loc, val*2)) # Twice after merging

                        self.tile.grids[below][x] *= 2
                        score += self.tile.grids[below][x]
                        self.tile.grids[y][x] = 0
                        below = -1

                        suc = True
                y -= 1

        # move all the grids down
        for x in range(self.tile.width):
            next_y = last_row
            for y in range(last_row, -1, -1):
                if self.tile.grids[y][x] == 0:
                    continue

                if next_y != y:
                    src_loc = Location(x, y)
                    dst_loc = Location(x, next_y)
                    val = self.tile.grids[y][x]
                    if src_loc in self.tile.animation_grids:
                        # update destination location
                        assert dst_loc not in self.tile.animation_grids
                        self.tile.animation_grids[dst_loc] = [
                            MovingGrid(grid.src_loc, grid.src_val, dst_loc, grid.dst_val) \
                                for grid in self.tile.animation_grids[src_loc] 
                        ]
                        del self.tile.animation_grids[src_loc]
                    else:
                        self.tile.animation_grids[dst_loc].append(MovingGrid(src_loc, val, dst_loc, val))

                    self.tile.grids[next_y][x] = self.tile.grids[y][x]
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
                    # Candidate to be merged
                    if left == -1 or self.tile.grids[y][x] != self.tile.grids[y][left]:
                        left = x
                    # Merged with the grid above
                    else:
                        src_loc = Location(x, y)
                        dst_loc = Location(left, y)
                        val = self.tile.grids[y][left]
                        self.tile.animation_grids[dst_loc].append(MovingGrid(dst_loc, val, dst_loc, 0)) # Disappear after merging
                        self.tile.animation_grids[dst_loc].append(MovingGrid(src_loc, val, dst_loc, val*2)) # Twice after merging

                        self.tile.grids[y][left] *= 2
                        score += self.tile.grids[y][left]
                        self.tile.grids[y][x] = 0
                        left = -1

                        suc = True
                x += 1

        # move all the grids left
        for y in range(self.tile.height):
            next_x = 0
            for x in range(self.tile.width):
                if self.tile.grids[y][x] == 0:
                    continue

                self.tile.grids[y][next_x] = self.tile.grids[y][x]

                if next_x != x:
                    src_loc = Location(x, y)
                    dst_loc = Location(next_x, y)
                    val = self.tile.grids[y][x]
                    if src_loc in self.tile.animation_grids:
                        # update destination location
                        assert dst_loc not in self.tile.animation_grids
                        self.tile.animation_grids[dst_loc] = [
                            MovingGrid(grid.src_loc, grid.src_val, dst_loc, grid.dst_val) \
                                for grid in self.tile.animation_grids[src_loc] 
                        ]
                        del self.tile.animation_grids[src_loc]
                    else:
                        self.tile.animation_grids[dst_loc].append(MovingGrid(src_loc, val, dst_loc, val))

                    self.tile.grids[y][next_x] = self.tile.grids[y][x]
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
                    # Candidate to be merged
                    if (
                        right == -1
                        or self.tile.grids[y][x] != self.tile.grids[y][right]
                    ):
                        right = x
                    # Merged with the grid above
                    else:
                        src_loc = Location(x, y)
                        dst_loc = Location(right, y)
                        val = self.tile.grids[y][right]
                        self.tile.animation_grids[dst_loc].append(MovingGrid(dst_loc, val, dst_loc, 0)) # Disappear after merging
                        self.tile.animation_grids[dst_loc].append(MovingGrid(src_loc, val, dst_loc, val*2)) # Twice after merging

                        self.tile.grids[y][right] *= 2
                        score += self.tile.grids[y][right]
                        self.tile.grids[y][x] = 0
                        right = -1

                        suc = True
                x -= 1

        # move all the grids right
        for y in range(self.tile.height):
            next_x = last_col
            for x in range(last_col, -1, -1):
                if self.tile.grids[y][x] == 0:
                    continue

                self.tile.grids[y][next_x] = self.tile.grids[y][x]

                if next_x != x:
                    src_loc = Location(x, y)
                    dst_loc = Location(next_x, y)
                    val = self.tile.grids[y][x]
                    if src_loc in self.tile.animation_grids:
                        # update destination location
                        assert dst_loc not in self.tile.animation_grids
                        self.tile.animation_grids[dst_loc] = [
                            MovingGrid(grid.src_loc, grid.src_val, dst_loc, grid.dst_val) \
                                for grid in self.tile.animation_grids[src_loc] 
                        ]
                        del self.tile.animation_grids[src_loc]
                    else:
                        self.tile.animation_grids[dst_loc].append(MovingGrid(src_loc, val, dst_loc, val))

                    self.tile.grids[y][next_x] = self.tile.grids[y][x]
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