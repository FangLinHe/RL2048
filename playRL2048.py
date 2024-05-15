#!/usr/bin/env python

import pygame

from RL2048.game_engine import GameEngine
from RL2048.tile import Tile
from RL2048.tile_plotter import PlotProperties, TilePlotter


def main():
    tile: Tile = Tile(width=4, height=4)
    plot_properties: PlotProperties = PlotProperties()
    plotter: TilePlotter = TilePlotter(tile, plot_properties)
    game_engine: GameEngine = GameEngine(tile)
    plotter.plot(game_engine.score)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_r:
                    game_engine.reset()
                if not game_engine.game_is_over:
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

                    plotter.plot(game_engine.score)
                    if game_engine.game_is_over:
                        print(f"Game over, score: {game_engine.score}")
                        plotter.plot_game_over()


if __name__ == "__main__":
    main()
