import pygame
from RL2048.tile import Tile
from typing import List, NamedTuple, Tuple


class PlotProperties(NamedTuple):
    grid_width: int = 100
    grid_height: int = 100
    grid_space: int = 10


class TilePlotter:
    def __init__(self, tile: Tile, plot_properties: PlotProperties):
        self.tile: Tile = tile
        self.plot_properties: PlotProperties = plot_properties

        pygame.init()

        self.win = pygame.display.set_mode((self.window_size()))
        self.rects: List[List[pygame.Rect]] = [
            [pygame.Rect(*self.grid_tlwh(x, y)) for x in range(self.tile.width)]
            for y in range(self.tile.height)
        ]

        self.font = pygame.font.SysFont("Comic Sans MS", 30)

    def window_size(self) -> Tuple[int, int]:
        height = (
            self.tile.height * self.plot_properties.grid_height
            + (self.tile.height + 1) * self.plot_properties.grid_space
        )
        width = (
            self.tile.width * self.plot_properties.grid_width
            + (self.tile.width + 1) * self.plot_properties.grid_space
        )

        return (width, height)

    def grid_left(self, x: int) -> int:
        left = (
            self.plot_properties.grid_space * (x + 1)
            + self.plot_properties.grid_width * x
        )

        return left

    def grid_top(self, y: int) -> int:
        top = (
            self.plot_properties.grid_space * (y + 1)
            + self.plot_properties.grid_height * y
        )

        return top

    def grid_tlwh(self, x, y) -> Tuple[int, int, int, int]:
        return (
            self.grid_left(x),
            self.grid_top(y),
            self.plot_properties.grid_width,
            self.plot_properties.grid_height,
        )

    def plot(self):
        for y, rects_row in enumerate(self.rects):
            for x, rect in enumerate(rects_row):
                pygame.draw.rect(self.win, (255, 128, 128), rect)
                grid_value = self.tile.grids[y][x]
                if grid_value != 0:
                    (grid_l, grid_t, grid_w, grid_h) = self.grid_tlwh(x, y)
                    text_surface = self.font.render(f"{grid_value}", True, (20, 20, 20))
                    text_rect = text_surface.get_rect(
                        center=(grid_l + grid_w / 2, grid_t + grid_h / 2)
                    )
                    self.win.blit(text_surface, text_rect)

        pygame.display.update()
