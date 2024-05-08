#!/usr/bin/env python

import pygame
from random import choice
from RL2048.game_engine import GameEngine
from RL2048.tile import Tile
from RL2048.tile_plotter import TilePlotter, PlotProperties
from typing import List


def main():
    tile: Tile = Tile(width=4, height=4)
    plot_properties: PlotProperties = PlotProperties()
    plotter: TilePlotter = TilePlotter(tile, plot_properties)
    game_engine: GameEngine = GameEngine(tile)
    plotter.plot(game_engine.score)

    keys = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]
    move_failures: List[int] = []
    scores: List[int] = []

    move_failure = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            
            if event.type == pygame.KEYUP and event.key == pygame.K_r:
                game_engine.reset()
            
        key = choice(keys)
        if not game_engine.game_is_over:
            if key == pygame.K_UP:
                move_result = game_engine.move_up()
                if move_result.suc:
                    game_engine.generate_new()
                else:
                    move_failure += 1
            elif key == pygame.K_DOWN:
                move_result = game_engine.move_down()
                if move_result.suc:
                    game_engine.generate_new()
                else:
                    move_failure += 1
            elif key == pygame.K_LEFT:
                move_result = game_engine.move_left()
                if move_result.suc:
                    game_engine.generate_new()
                else:
                    move_failure += 1
            elif key == pygame.K_RIGHT:
                move_result = game_engine.move_right()
                if move_result.suc:
                    game_engine.generate_new()
                else:
                    move_failure += 1

            plotter.plot(game_engine.score)
            if game_engine.game_is_over:
                plotter.plot_game_over()
                move_failures.append(move_failure)
                move_failure = 0
                scores.append(game_engine.score)
                print(f"Move failures: {move_failures}")
                print(f"Scores: {scores}")
                game_engine.reset()



if __name__ == "__main__":
    main()
