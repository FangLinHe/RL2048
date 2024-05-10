#!/usr/bin/env python

import argparse
import json
import pygame
import time
import math

from datetime import datetime
from random import choice
from RL2048.game_engine import GameEngine, MoveResult
from RL2048.tile import Tile
from RL2048.tile_plotter import TilePlotter, PlotProperties
from RL2048.DQN.dqn import TrainingParameters, DQN
from RL2048.DQN.net import Net
from RL2048.DQN.replay_memory import Action, Transition
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
        default="Experiments/DQN",
        help="Prefix of output json file",
    )
    parser.add_argument(
        "--max_iters",
        default=1000,
        type=int,
        help="Max iterations of experiments; set it to negative value to run infinitely",
    )
    args = parser.parse_args()

    return args


def make_state(tile: Tile) -> Sequence[float]:
    def one_hot(v: int, size: int = 16) -> Sequence[int]:
        loc = int(math.log2(float(v))) if v > 0 else 0
        return [1 if i == loc else 0 for i in range(size)]

    one_hot_grid = [one_hot(value) for row in tile.grids for value in row]
    return [float(one_hot_value) for row in one_hot_grid for one_hot_value in row]


INVALID_MOVEMENT_REWARD: float = -16
GAME_OVER_REWARD: float = -128

def main(show_board: bool, print_results: bool, output_prefix: str, max_iters: int):
    tile: Tile = Tile(width=4, height=4)
    plot_properties: PlotProperties = PlotProperties()
    plotter: TilePlotter = TilePlotter(tile, plot_properties)
    game_engine: GameEngine = GameEngine(tile)

    move_failures: List[int] = []
    total_scores: List[int] = []
    max_grids: List[int] = []

    # DQN part
    in_features: int = tile.width * tile.height * 16
    out_features: int = len(Action)
    hidden_layers: List[int] = [256, 128, 64]
    policy_net = Net(in_features, out_features, hidden_layers)
    training_params = TrainingParameters(
        memory_capacity=20000, gamma=0.99, batch_size=64, lr=0.000001,
    )
    dqn = DQN(policy_net, training_params)

    move_failure = 0
    date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_fn = f"{output_prefix}_{date_time_str}.json"

    iter = 0
    start_time = time.time()
    cur_state: Sequence[int] = make_state(tile)
    next_state: Sequence[int] = []
    total_rewards: List[float] = []
    total_reward: float = 0.0
    while iter < max_iters:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYUP and event.key == pygame.K_r:
                game_engine.reset()

        if not game_engine.game_is_over:
            action: Action = dqn.get_action_epsilon_greedy(cur_state)
            reward: float = 0.0
            if action == Action.UP:
                move_result = game_engine.move_up()
            elif action == Action.DOWN:
                move_result = game_engine.move_down()
            elif action == Action.LEFT:
                move_result = game_engine.move_left()
            else:  # action == Action.RIGHT
                move_result = game_engine.move_right()

            if move_result.suc:
                game_engine.generate_new()
                reward += move_result.score
            else:
                move_failure += 1
                reward += INVALID_MOVEMENT_REWARD

            if game_engine.game_is_over:
                reward += GAME_OVER_REWARD  # + tile.max_grid())

            next_state = make_state(tile)
            total_reward += reward

            transition = Transition(
                state=cur_state,
                action=action,
                next_state=next_state,
                reward=reward,
                game_over=game_engine.game_is_over
            )

            if game_engine.game_is_over:
                total_rewards.append(total_reward)
                total_reward = 0.0
            
            dqn.push_transition_and_optimize_automatically(transition)

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
                    print(f"total_rewards: {total_rewards}")

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
