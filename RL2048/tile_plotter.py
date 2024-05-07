import pygame
import RL2048.colors as colors
from RL2048.common import Location
from RL2048.tile import Tile
from typing import Dict, List, NamedTuple, Tuple


class PlotProperties(NamedTuple):
    grid_width: int = 100
    grid_height: int = 100
    grid_space: int = 10
    border_radius: int = 3
    animation_steps: int = 10


class Animation:
    def __init__(self, old_value: int, src: Location, dst: Location, step: int = 10):
        self.old_value = old_value
        self.src = src
        self.dst = dst
        self.diff = Location(dst.x - src.x, dst.y - src.y)
        self.step = step
        self.count = 0

    def next_location(self) -> Location:
        if self.count > self.step:
            return self.dst
        move_ratio = self.count / (self.step + 1)
        new_location = Location(
            self.src.x + round(self.diff.x * move_ratio),
            self.src.y + round(self.diff.y * move_ratio),
        )
        self.count += 1
        return new_location


class TilePlotter:
    def __init__(self, tile: Tile, plot_properties: PlotProperties):
        self.tile: Tile = tile
        self.plot_properties: PlotProperties = plot_properties

        pygame.init()

        self.win = pygame.display.set_mode((self.window_size()))
        self.win.fill(colors.win_background_color)
        self.clock = pygame.time.Clock()
        self.rects: List[List[pygame.Rect]] = [
            [pygame.Rect(*self.grid_tlwh(x, y)) for x in range(self.tile.width)]
            for y in range(self.tile.height)
        ]

        self.font = pygame.font.SysFont("Comic Sans MS", 35)

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
                grid_value = self.tile.grids[y][x]
                colorset = (
                    colors.color_palette[grid_value]
                    if grid_value in colors.color_palette
                    else colors.default_colorset
                )
                pygame.draw.rect(
                    self.win,
                    tuple(colorset.background),
                    rect,
                    border_radius=self.plot_properties.border_radius,
                )
                if grid_value != 0:
                    (grid_l, grid_t, grid_w, grid_h) = self.grid_tlwh(x, y)
                    text_surface = self.font.render(
                        f"{grid_value}", True, colorset.foreground
                    )
                    text_rect = text_surface.get_rect(
                        center=(grid_l + grid_w / 2, grid_t + grid_h / 2)
                    )
                    self.win.blit(text_surface, text_rect)

        pygame.display.update()
