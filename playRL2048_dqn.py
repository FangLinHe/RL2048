#!/usr/bin/env python

import argparse
import json
import pygame
import time

from datetime import datetime
from random import choice
from RL2048.game_engine import GameEngine
from RL2048.tile import Tile
from RL2048.tile_plotter import TilePlotter, PlotProperties
from RL2048.DQN.dqn import Action, TrainingParameters, DQN, Net
from typing import List, Sequence, Optional

from RL2048.DQN.dqn import DQN


def parse_args():
    parser = argparse.ArgumentParser(
        prog="PlayRL2048Random", description="Play 2048 with random actions"
    )
    parser.add_argument(
        "--show_board", action="store_true", help="Show the procedure of the game"
    )
    parser.add_argument(
        "--print_results",
        action="store_true",
        help="Print results, like scores, failure times, etc.",
    )
    parser.add_argument(
        "--output_prefix",
        default="Experiments/random",
        help="Prefix of output json file",
    )
    parser.add_argument(
        "--max_iters",
        default=100,
        type=int,
        help="Max iterations of experiments; set it to negative value to run infinitely",
    )
    args = parser.parse_args()

    return args


def make_state(tile: Tile) -> Sequence[int]:
    return [value for row in tile.grids for value in row]


def main(show_board: bool, print_results: bool, output_prefix: str, max_iters: int):
    tile: Tile = Tile(width=4, height=4)
    plot_properties: PlotProperties = PlotProperties()
    plotter: TilePlotter = TilePlotter(tile, plot_properties)
    game_engine: GameEngine = GameEngine(tile)

    keys = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]
    move_failures: List[int] = []
    total_scores: List[int] = []
    max_grids: List[int] = []

    # DQN part
    in_features: int = tile.width * tile.height
    out_features: int = len(Action)
    hidden_layers: List[int] = [64, 48, 32]
    policy_net = Net(in_features, out_features, hidden_layers)
    training_params = TrainingParameters(
        memory_capacity=1024, gamma=0.99, batch_size=64, lr=0.001
    )
    dqn = DQN(policy_net, training_params)

    move_failure = 0
    date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_fn = f"{output_prefix}_{date_time_str}.json"

    iter = 0
    start_time = time.time()
    cur_state: Optional[Sequence[int]] = None
    while iter < max_iters:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYUP and event.key == pygame.K_r:
                game_engine.reset()

        if not game_engine.game_is_over:
            cur_state = make_state(tile)
            next_action: Action = dqn.next_action_epsilon_greedy(cur_state)
            if next_action == Action.UP:
                move_result = game_engine.move_up()
                if move_result.suc:
                    game_engine.generate_new()
                else:
                    move_failure += 1
            elif next_action == Action.DOWN:
                move_result = game_engine.move_down()
                if move_result.suc:
                    game_engine.generate_new()
                else:
                    move_failure += 1
            elif next_action == Action.LEFT:
                move_result = game_engine.move_left()
                if move_result.suc:
                    game_engine.generate_new()
                else:
                    move_failure += 1
            elif next_action == Action.RIGHT:
                move_result = game_engine.move_right()
                if move_result.suc:
                    game_engine.generate_new()
                else:
                    move_failure += 1

            if show_board:
                plotter.plot(game_engine.score)
            else:
                tile.reset_animation_grids()
            if game_engine.game_is_over:
                iter += 1
                if show_board:
                    plotter.plot_game_over()
                move_failures.append(move_failure)
                move_failure = 0
                max_grids.append(tile.max_grid())
                total_scores.append(game_engine.score)
                if print_results:
                    print(f"Move failures: {move_failures}")
                    print(f"Total scores: {total_scores}")
                    print(f"Max grids: {max_grids}")

                output_json = {
                    "move_failures": move_failures,
                    "total_scores": total_scores,
                    "max_grids": max_grids,
                }
                with open(output_json_fn, "w") as fid:
                    json.dump(output_json, fid)
                game_engine.reset()

    elapsed_sec = time.time() - start_time
    print(
        f"Done running {max_iters} times of experiments in {round(elapsed_sec * 1000.0)} millisecond(s)."
    )
    print(f"See results in {output_json_fn}.")


if __name__ == "__main__":
    args = parse_args()
    main(args.show_board, args.print_results, args.output_prefix, args.max_iters)
