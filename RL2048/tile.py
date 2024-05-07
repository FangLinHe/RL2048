from random import randint, sample
from typing import List


class Tile:
    def __init__(self, width: int = 4, height: int = 4):
        self.width: int = width
        self.height: int = height
        self.grids: List[List[int]] = []
        self.random_start()

    def random_start(self):
        self.grids = [[0 for _x in range(self.width)] for _y in range(self.height)]

        def fill_grid(index):
            x, y = index % self.width, index // self.width
            self.grids[y][x] = 2 ** randint(1, 2)

        random_count = 2
        random_indices = sample(range(self.width * self.height), random_count)
        assert len(set(random_indices)) == random_count
        for index in random_indices:
            fill_grid(index)
