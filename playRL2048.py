#!/usr/bin/env python

import pygame
from RL2048.game_engine import GameEngine
from RL2048.tile import Tile
from RL2048.tile_plotter import TilePlotter, PlotProperties


def main():
    tile: Tile = Tile(width=4, height=4)
    plot_properties: PlotProperties = PlotProperties()
    plotter: TilePlotter = TilePlotter(tile, plot_properties)
    game_engine: GameEngine = GameEngine(tile)
    plotter.plot()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    print(f"Move up suc: {game_engine.move_up()}")
                elif event.key == pygame.K_DOWN:
                    print(f"Move down suc: {game_engine.move_down()}")
                elif event.key == pygame.K_LEFT:
                    print(f"Move left suc: {game_engine.move_left()}")
                elif event.key == pygame.K_RIGHT:
                    print(f"Move right suc: {game_engine.move_right()}")
                
                plotter.plot()


if __name__ == "__main__":
    main()
