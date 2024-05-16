from typing import List

from rl_2048.tile import Tile


def test_tile():
    tile = Tile(width=5, height=3)
    for _ in range(100):
        assert tile.height == 3 and len(tile.grids) == 3
        assert tile.width == 5 and all(len(row) == 5 for row in tile.grids)

        non_zero_grids: List[int] = [g for row in tile.grids for g in row if g > 0]
        assert len(non_zero_grids) == tile.random_start_count
        assert all(g in {2, 4} for g in non_zero_grids)
        assert tile.max_grid() == max(non_zero_grids)
        assert len(tile.animation_grids) == 2
        tile.reset_animation_grids()
        assert len(tile.animation_grids) == 0
        tile.random_start()
