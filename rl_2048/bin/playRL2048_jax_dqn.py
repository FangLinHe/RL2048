#!/usr/bin/env python

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from random import shuffle
from typing import Callable

import jax
import pygame
from flax.training.checkpoints import PyTree, restore_checkpoint
from jax import Array
from jax import random as jrandom

from rl_2048.dqn.common import Action, DQNParameters
from rl_2048.dqn.jax.dqn import (
    DQN,
    TrainingParameters,
)
from rl_2048.dqn.jax.net import PREDEFINED_NETWORKS, load_predefined_net
from rl_2048.dqn.jax.utils import flat_one_hot
from rl_2048.dqn.replay_memory import Transition
from rl_2048.game_engine import GameEngine, MoveResult
from rl_2048.tile import Tile
from rl_2048.tile_plotter import PlotProperties, TilePlotter


def parse_args():
    parser = argparse.ArgumentParser(
        prog="PlayRL2048Random", description="Play 2048 with random actions"
    )
    parser.add_argument(
        "--show_board",
        action="store_true",
        help="Show the procedure of the game",
    )
    parser.add_argument(
        "--print_results",
        action="store_true",
        help="Print results, like scores, failure times, etc.",
    )
    parser.add_argument(
        "--output_json_prefix",
        default="Experiments/jax_dqn",
        help="Prefix of output json file",
    )
    parser.add_argument(
        "--output_net_prefix",
        default="TrainedNetworks/jax_dqn",
        help="Prefix of output json file",
    )
    parser.add_argument(
        "--max_iters",
        default=1000,
        type=int,
        help="Max iterations of experiments; set it to negative value to run infinitely",
    )
    parser.add_argument(
        "--trained_net_path",
        type=str,
        default="",
        help="Path to pre-trained or trained network to train / play DQN. If in train mode, "
        "the weights are initialized from pre-trained model.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Train mode or eval mode. If eval mode, --trained_net_path must be specified.",
    )
    parser.add_argument(
        "--network_version",
        default="layers_1024_512_256",
        type=str,
        help="Network version which maps to certain network structure. "
        f"Should be in {PREDEFINED_NETWORKS}",
    )
    args = parser.parse_args()

    if args.eval and args.trained_net_path == "":
        raise ValueError(
            "`--eval` is specified, so `--trained_net_path` must be specified"
        )

    return args


def make_state_from_grids(tile: Tile) -> Sequence[float]:
    return [float(value) for row in tile.grids for value in row]


INVALID_MOVEMENT_REWARD: float = -(2**8)
GAME_OVER_REWARD: float = -(2**12)


def write_json(move_failures, total_scores, max_grids, total_rewards, filepath: str):
    output_json = {
        "move_failures": move_failures,
        "total_scores": total_scores,
        "max_grids": max_grids,
        "total_rewards": total_rewards,
    }
    tmp_file = f"{filepath}.tmp"
    with open(tmp_file, "w") as fid:
        json.dump(output_json, fid)
        shutil.move(tmp_file, filepath)


def eval_dqn(
    show_board: bool,
    print_results: bool,
    output_json_prefix: str,
    max_iters: int,
    trained_net_path: str,
    network_version: str,
):
    tile: Tile = Tile(width=4, height=4)
    plot_properties: PlotProperties = PlotProperties(fps=60, delay_after_plot=50)
    plotter: TilePlotter = TilePlotter(tile, plot_properties)
    game_engine: GameEngine = GameEngine(tile)

    move_failures: list[int] = []
    total_scores: list[int] = []
    max_grids: list[int] = []

    # DQN part
    out_features: int = len(Action)
    policy_net = load_predefined_net(
        network_version,
        out_features,
    )
    policy_net_apply: Callable = jax.jit(policy_net.apply)
    policy_net_variables: PyTree
    try:
        policy_net_variables = restore_checkpoint(
            os.path.dirname(trained_net_path), None
        )
    except FileNotFoundError:
        try:
            print(f"Loading model from {trained_net_path} failed.")
            if "rl_2048/" in trained_net_path:
                search_in_parent_path: str = trained_net_path.replace("rl_2048/", "")
                print(f"Try to load model from {search_in_parent_path}")
                policy_net_variables = restore_checkpoint(
                    os.path.dirname(search_in_parent_path), None
                )
            else:
                raise
        except FileNotFoundError:
            print(f"Loading model from {search_in_parent_path} still failed.")
            raise
    policy_net_variables = {
        "batch_stats": policy_net_variables["batch_stats"],
        "params": policy_net_variables["params"],
    }
    move_failure = 0
    date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_fn = f"{output_json_prefix}_{date_time_str}.json"

    iter = 0
    start_time = time.time()
    cur_state: Sequence[float] = flat_one_hot(tile.flattened(), 16)

    action_candidates: list[Action] = []
    inf_times = []

    prev_score: int = game_engine.score
    score_not_increasing_count: int = 0
    while max_iters < 0 or iter < max_iters:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYUP and event.key == pygame.K_r:
                game_engine.reset()
                prev_score = game_engine.score

        if not game_engine.game_is_over:
            start_inf_time = time.time()
            policy_net_output = DQN.infer_action_net(
                policy_net_apply, policy_net_variables, cur_state
            )
            inf_times.append(time.time() - start_inf_time)

            action: Action = policy_net_output.action

            move_result: MoveResult
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
            else:
                move_failure += 1
                action_candidates = [
                    candidate for candidate in Action if candidate != action
                ]
                shuffle(action_candidates)
                for action in action_candidates:
                    if action == Action.UP:
                        move_result = game_engine.move_up()
                    elif action == Action.DOWN:
                        move_result = game_engine.move_down()
                    elif action == Action.LEFT:
                        move_result = game_engine.move_left()
                    else:  # action == Action.RIGHT
                        move_result = game_engine.move_right()

                    if move_result.suc:
                        break
                    move_failure += 1
                if not move_result.suc:
                    raise ValueError("Game is not over yet but all actions failed")

            if game_engine.score == prev_score:
                score_not_increasing_count += 1
            else:
                score_not_increasing_count = 0
            prev_score = game_engine.score

            cur_state = flat_one_hot(tile.flattened(), 16)

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
                if iter % 10 == 0:
                    write_json(
                        move_failures,
                        total_scores,
                        max_grids,
                        [],  # total_rewards,
                        output_json_fn,
                    )
                game_engine.reset()

    write_json(move_failures, total_scores, max_grids, [], output_json_fn)
    elapsed_sec = time.time() - start_time
    print(
        f"Done running {max_iters} times of experiments in {round(elapsed_sec * 1000.0)} millisecond(s)."
    )
    print(
        f"Average inference time: {(sum(inf_times) / len(inf_times)) * 1000.0} millisecond(s)"
    )
    print(f"See results in {output_json_fn}.")


def train(
    show_board: bool,
    print_results: bool,
    output_json_prefix: str,
    output_net_prefix: str,
    max_iters: int,
    network_version: str,
    pretrained_net_path: str = "",
):
    tile: Tile = Tile(width=4, height=4)
    plot_properties: PlotProperties = PlotProperties(fps=60, delay_after_plot=50)
    plotter: TilePlotter = TilePlotter(tile, plot_properties)
    game_engine: GameEngine = GameEngine(tile)

    move_failures: list[int] = []
    total_scores: list[int] = []
    max_grids: list[int] = []

    # DQN part
    in_features: int = tile.width * tile.height * 16
    out_features: int = len(Action)
    policy_net = load_predefined_net(
        network_version,
        out_features,
    )

    dqn_parameters = DQNParameters(
        memory_capacity=20000,
        batch_size=128,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=10000,
    )
    training_params = TrainingParameters(
        gamma=0.99,
        batch_size=128,
        optimizer="adamw",
        lr=1e-4,
        lr_decay_milestones=[],
        lr_gamma=1.0,
        loss_fn="huber_loss",
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=10000,
        TAU=0.005,
        save_network_steps=2000,
        print_loss_steps=500,
        tb_write_steps=50,
        pretrained_net_path=pretrained_net_path,
    )
    reward_norm_factor: int = 256  # reward / reward_norm_factor for value function
    rng: Array = jrandom.key(0)

    move_failure = 0
    date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_fn = f"{output_json_prefix}_{date_time_str}.json"
    output_net_dir = (
        f"{output_net_prefix}/{os.path.basename(output_json_prefix)}_{date_time_str}"
    )
    os.makedirs(output_net_dir)

    dqn = DQN(
        in_features, policy_net, output_net_dir, dqn_parameters, training_params, rng
    )
    if pretrained_net_path != "":
        dqn.load_model(pretrained_net_path)

    dqn.summary_writer.add_text("output_json_path", output_json_fn)

    iter = 0
    start_time = time.time()
    cur_state: Sequence[float] = flat_one_hot(tile.flattened(), 16)
    next_state: Sequence[float] = cur_state
    total_rewards: list[float] = []
    suc_move_statistics: dict[Action, int] = defaultdict(int)
    total_reward: float = 0.0

    new_collect_count: int = 0
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
                suc_move_statistics[action] += 1
                game_engine.generate_new()
                reward += move_result.score
            else:
                move_failure += 1
                reward += INVALID_MOVEMENT_REWARD

            if game_engine.game_is_over:
                reward += GAME_OVER_REWARD

            next_state = flat_one_hot(tile.flattened(), 16)
            total_reward += reward

            transition = Transition(
                state=cur_state,
                action=action,
                next_state=next_state,
                reward=reward / reward_norm_factor,
                game_over=game_engine.game_is_over,
            )
            cur_state = next_state[:]

            dqn.push_transition(transition)
            new_collect_count += 1
            if new_collect_count >= training_params.batch_size:
                _loss = dqn.optimize_model(game_iter=iter)
                new_collect_count = 0

            if show_board:
                plotter.plot(game_engine.score)
            elif move_result.suc:
                tile.reset_animation_grids()
            if print_results:
                print(".\r")
            if game_engine.game_is_over:
                iter += 1
                if show_board:
                    plotter.plot_game_over()
                move_failures.append(move_failure)
                max_grid: int = tile.max_grid()
                max_grids.append(max_grid)
                total_scores.append(game_engine.score)

                dqn.summary_writer.add_scalar(
                    "game_statistics/opt_steps", dqn.optimize_steps, iter
                )
                dqn.summary_writer.add_scalar(
                    "game_statistics/move_failures", move_failure, iter
                )
                dqn.summary_writer.add_scalar(
                    "game_statistics/total_scores",
                    game_engine.score,
                    iter,
                )
                dqn.summary_writer.add_scalar(
                    "game_statistics/max_grids", max_grid, iter
                )
                dqn.summary_writer.add_scalar(
                    "game_statistics/total_rewards", total_reward, iter
                )

                total_rewards.append(total_reward)
                total_reward = 0.0

                if print_results:
                    print(f"Move failures: {move_failures}")
                    print(f"Total scores: {total_scores}")
                    print(f"Max grids: {max_grids}")
                    print(f"total_rewards: {total_rewards}")

                if iter % 10 == 0:
                    write_json(
                        move_failures,
                        total_scores,
                        max_grids,
                        total_rewards,
                        output_json_fn,
                    )
                move_failure = 0
                game_engine.reset()

    write_json(move_failures, total_scores, max_grids, total_rewards, output_json_fn)
    elapsed_sec = time.time() - start_time
    print(
        f"Done running {max_iters} times of experiments in {round(elapsed_sec * 1000.0)} millisecond(s)."
    )
    print(f"See results in {output_json_fn}.")


def main():
    args = parse_args()
    if args.eval:
        eval_dqn(
            args.show_board,
            args.print_results,
            args.output_json_prefix,
            args.max_iters,
            args.trained_net_path,
            args.network_version,
        )
    else:
        train(
            args.show_board,
            args.print_results,
            args.output_json_prefix,
            args.output_net_prefix,
            args.max_iters,
            args.network_version,
            args.trained_net_path,
        )


if __name__ == "__main__":
    main()
