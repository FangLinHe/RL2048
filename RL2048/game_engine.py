from RL2048.tile import Tile

from typing import List, Tuple, NamedTuple

class Location(NamedTuple):
    x: int
    y: int

class GameEngine:
    def __init__(self, tile: Tile):
        self.tile = tile
    
    def move_up(self) -> bool:
        suc: bool = False

        # move the up-most grids to the first row
        for x in range(self.tile.width):
            for y in range(self.tile.height):
                if self.tile.grids[y][x] > 0:
                    # Do nothing, already the up-most grid
                    if y == 0:
                        break

                    assert self.tile.grids[0][x] == 0
                    self.tile.grids[0][x] = self.tile.grids[y][x]
                    self.tile.grids[y][x] = 0
                    suc = True
                    break

        # merge grids up if values are the same as the grid above
        # and values are added up

        return suc
    
    def move_down(self) -> bool:
        suc: bool = False

        last_row: int = self.tile.height - 1
        # move the bottom-most grids to the last row
        for x in range(self.tile.width):
            for y in range(last_row, -1, -1):
                if self.tile.grids[y][x] > 0:
                    # Do nothing, already the bottom-most grid
                    if y == last_row:
                        break

                    assert self.tile.grids[last_row][x] == 0
                    self.tile.grids[last_row][x] = self.tile.grids[y][x]
                    self.tile.grids[y][x] = 0
                    suc = True
                    break

        # merge grids up if values are the same as the grid below
        # and values are added up

        return suc
    
    def move_left(self) -> bool:
        suc: bool = False

        # move the left-most grids to the first column
        for y in range(self.tile.height):
            for x in range(self.tile.width):
                if self.tile.grids[y][x] > 0:
                    # Do nothing, already the left-most grid
                    if x == 0:
                        break

                    assert self.tile.grids[y][0] == 0
                    self.tile.grids[y][0] = self.tile.grids[y][x]
                    self.tile.grids[y][x] = 0
                    suc = True
                    break

        # merge grids up if values are the same as the left grid
        # and values are added up

        return suc
    
    def move_right(self) -> bool:
        suc: bool = False

        last_column: int = self.tile.width - 1
        # move the right-most grids to the last column
        for y in range(self.tile.height):
            for x in range(last_column, -1, -1):
                if self.tile.grids[y][x] > 0:
                    # Do nothing, already the right-most grid
                    if x == last_column:
                        break

                    assert self.tile.grids[y][last_column] == 0
                    self.tile.grids[y][last_column] = self.tile.grids[y][x]
                    self.tile.grids[y][x] = 0
                    suc = True
                    break

        # merge grids up if values are the same as the left grid
        # and values are added up

        return suc