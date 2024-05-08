import pygame
import RL2048.colors as colors
from RL2048.common import Location
from RL2048.tile import MovingGrid, Tile
from collections import defaultdict
from typing import Dict, List, NamedTuple, Tuple


class PlotProperties(NamedTuple):
    info_board_height: int = 110
    grid_width: int = 100
    grid_height: int = 100
    grid_space: int = 10
    border_radius: int = 3
    animation_steps: int = 8
    fps: int = 120


class Animation:
    def __init__(
        self, grid: MovingGrid, src_coord: Location, dst_coord: Location, step: int = 10
    ):
        self.grid = grid
        self.src_coord = src_coord
        self.dst_coord = dst_coord
        self.diff = Location(dst_coord.x - src_coord.x, dst_coord.y - src_coord.y)
        self.step = step
        self.count = 0

    def next_location(self) -> Location:
        if self.count >= self.step:
            return self.dst_coord
        move_ratio = self.count / (self.step - 1)
        new_location = Location(
            self.src_coord.x + round(self.diff.x * move_ratio),
            self.src_coord.y + round(self.diff.y * move_ratio),
        )
        self.count += 1
        return new_location


class TilePlotter:
    def __init__(self, tile: Tile, plot_properties: PlotProperties):
        self.tile: Tile = tile
        self.plot_properties: PlotProperties = plot_properties

        pygame.init()
        self.clock = pygame.time.Clock()

        self.win = pygame.display.set_mode((self.window_size()))
        self.game_surface = pygame.surface.Surface(self.game_surface_size())
        self.rects: List[List[pygame.Rect]] = [
            [pygame.Rect(*self.grid_tlwh(x, y)) for x in range(self.tile.width)]
            for y in range(self.tile.height)
        ]

        self.score_font = pygame.font.SysFont("Comic Sans MS", 20)
        self.font = pygame.font.SysFont("Comic Sans MS", 35)

        self.clean_canvas()

    def window_size(self) -> Tuple[int, int]:
        game_surface_w, game_surface_h = self.game_surface_size()

        return (game_surface_w, game_surface_h + self.plot_properties.info_board_height)

    def game_surface_size(self) -> Tuple[int, int]:
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

    def grid_tl(self, xy: Location) -> Location:
        return Location(self.grid_left(xy.x), self.grid_top(xy.y))

    def grid_tlwh(self, x, y) -> Tuple[int, int, int, int]:
        return (
            self.grid_left(x),
            self.grid_top(y),
            self.plot_properties.grid_width,
            self.plot_properties.grid_height,
        )

    def plot_grid(
        self, grid_value: int, rect: pygame.Rect, tlwh: Tuple[int, int, int, int]
    ):
        colorset = (
            colors.color_palette[grid_value]
            if grid_value in colors.color_palette
            else colors.default_colorset
        )
        pygame.draw.rect(
            self.game_surface,
            tuple(colorset.background),
            rect,
            border_radius=self.plot_properties.border_radius,
        )
        if grid_value != 0:
            grid_l, grid_t, grid_w, grid_h = tlwh
            text_surface = self.font.render(f"{grid_value}", True, colorset.foreground)
            text_rect = text_surface.get_rect(
                center=(grid_l + grid_w / 2, grid_t + grid_h / 2)
            )
            self.game_surface.blit(text_surface, text_rect)

    def clean_canvas(self):
        self.win.fill(colors.info_board_color)
        text_surface = self.score_font.render(f"Score", True, colors.dark_foreground)
        text_rect = text_surface.get_rect(
            center=(self.win.get_width() // 2, 20)
        )
        self.win.blit(text_surface, text_rect)

    def plot_score(self, score):
        text_surface = self.font.render(f"{score}", True, colors.dark_foreground)
        text_rect = text_surface.get_rect(
            center=(self.win.get_width() // 2, self.plot_properties.info_board_height // 2)
        )
        self.win.blit(text_surface, text_rect)


    def plot(self, score: int):
        animations = [
            Animation(
                grid,
                self.grid_tl(grid.src_loc),
                self.grid_tl(grid.dst_loc),
                self.plot_properties.animation_steps,
            )
            for grids in self.tile.animation_grids.values()
            for grid in grids
        ]
        moving_locations = {a.grid.dst_loc for a in animations if a.grid.dst_loc != a.grid.src_loc}
        self.tile.animation_grids = defaultdict(list)

        src_rects = [
            pygame.Rect(*self.grid_tlwh(*animation.grid.src_loc))
            for animation in animations
        ]
        grid_w, grid_h = (
            self.plot_properties.grid_width,
            self.plot_properties.grid_height,
        )

        if len(moving_locations) > 0:
            self.plot_score(score)
            steps: int = self.plot_properties.animation_steps
            for t in range(steps):
                # Plot non-moving grids
                self.game_surface.fill(colors.win_background_color)
                for y, rects_row in enumerate(self.rects):
                    for x, rect in enumerate(rects_row):
                        if Location(x, y) not in moving_locations:
                            grid_value = self.tile.grids[y][x]
                            self.plot_grid(grid_value, rect, self.grid_tlwh(x, y))
                        else:
                            self.plot_grid(0, rect, self.grid_tlwh(x, y))
                self.win.blit(self.game_surface, (0, self.plot_properties.info_board_height))
                pygame.display.update()

                # Plot moving grids
                affected_areas: List[pygame.Rect] = []
                for animation, rect in zip(animations, src_rects):
                    loc = animation.next_location()
                    tlwh = (*loc, grid_w, grid_h)
                    dx = loc.x - rect.x
                    dy = loc.y - rect.y
                    affected_areas.append(pygame.Rect(
                        rect.left if dx > 0 else rect.left + dx,
                        self.plot_properties.info_board_height + (rect.top if dy > 0 else rect.top + dy),
                        rect.width + abs(dx),
                        rect.height + abs(dy)
                    ))
                    rect.move_ip(dx, dy)
                    val = animation.grid.dst_val if t == steps - 1 else animation.grid.src_val
                    self.plot_grid(val, rect, tlwh)
                self.win.blit(self.game_surface, (0, self.plot_properties.info_board_height))
                pygame.display.update(affected_areas)
                self.clock.tick(self.plot_properties.fps)


        # Plot all grids after the animation
        self.clean_canvas()
        self.game_surface.fill(colors.win_background_color)
        for y, rects_row in enumerate(self.rects):
            for x, rect in enumerate(rects_row):
                grid_value = self.tile.grids[y][x]
                self.plot_grid(grid_value, rect, self.grid_tlwh(x, y))
        self.win.blit(self.game_surface, (0, self.plot_properties.info_board_height))
        self.plot_score(score)
        pygame.display.update()
        self.clock.tick(self.plot_properties.fps)
