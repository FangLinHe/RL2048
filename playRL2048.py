#!/usr/bin/env python

import pygame
from RL2048.tile import Tile
from RL2048.tile_plotter import TilePlotter, PlotProperties


def main():
    tile: Tile = Tile(width=4, height=4)
    plot_properties: PlotProperties = PlotProperties()
    plotter: TilePlotter = TilePlotter(tile, plot_properties)
    plotter.plot()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()


if __name__ == "__main__":
    main()
