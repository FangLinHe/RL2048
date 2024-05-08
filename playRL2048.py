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
                    move_result = game_engine.move_up()
                    if move_result.suc:
                        game_engine.generate_new()
                    print(f"Move up result: {move_result}")
                elif event.key == pygame.K_DOWN:
                    move_result = game_engine.move_down()
                    if move_result.suc:
                        game_engine.generate_new()
                    print(f"Move down result: {move_result}")
                elif event.key == pygame.K_LEFT:
                    move_result = game_engine.move_left()
                    if move_result.suc:
                        game_engine.generate_new()
                    print(f"Move left result: {move_result}")
                elif event.key == pygame.K_RIGHT:
                    move_result = game_engine.move_right()
                    if move_result.suc:
                        game_engine.generate_new()
                    print(f"Move right result: {move_result}")
                elif event.key == pygame.K_r:
                    game_engine.reset()

                plotter.plot()


if __name__ == "__main__":
    main()
