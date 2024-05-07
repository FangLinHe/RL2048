import pygame
from RL2048.tile import Tile
from typing import Dict, List, NamedTuple, Tuple


class Color(NamedTuple):
    r: int
    g: int
    b: int


class ColorSet(NamedTuple):
    background: Color
    foreground: Color


light_foreground: Color = Color(248, 246, 242)
dark_foreground: Color = Color(117, 110, 102)

default_colorset = ColorSet(Color(128, 128, 128), light_foreground)
win_background_color: Color = Color(185, 173, 161)

color_palette: Dict[int, ColorSet] = {
    0: ColorSet(Color(202, 193, 181), dark_foreground),
    2: ColorSet(Color(236, 228, 219), dark_foreground),
    4: ColorSet(Color(236, 225, 204), dark_foreground),
    8: ColorSet(Color(233, 181, 130), light_foreground),
    16: ColorSet(Color(233, 154, 109), light_foreground),
    32: ColorSet(Color(231, 131, 103), light_foreground),
    64: ColorSet(Color(229, 105, 72), light_foreground),
    128: ColorSet(Color(232, 209, 128), light_foreground),
    256: ColorSet(Color(232, 205, 114), light_foreground),
    512: ColorSet(Color(231, 202, 101), light_foreground),
}


class PlotProperties(NamedTuple):
    grid_width: int = 100
    grid_height: int = 100
    grid_space: int = 10
    border_radius: int = 3


class TilePlotter:
    def __init__(self, tile: Tile, plot_properties: PlotProperties):
        self.tile: Tile = tile
        self.plot_properties: PlotProperties = plot_properties

        pygame.init()

        self.win = pygame.display.set_mode((self.window_size()))
        self.win.fill(win_background_color)
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
                    color_palette[grid_value]
                    if grid_value in color_palette
                    else default_colorset
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
