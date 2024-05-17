#!/usr/bin/env python

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Sequence, Set, Tuple

import pygame
from flax import linen as nn
from jax import Array
from jax import random as jrandom

from rl_2048.game_engine import GameEngine
from rl_2048.jax_dqn.dqn import DQN, TrainingParameters
from rl_2048.jax_dqn.net import Net
from rl_2048.jax_dqn.replay_memory import Action, Transition
from rl_2048.jax_dqn.utils import flat_one_hot
from rl_2048.tile import Tile
from rl_2048.tile_plotter import PlotProperties, TilePlotter

PREDEFINED_NETWORKS: Set[str] = {
    "layers_1024_512_256",
    # "layers_512_512_residual_0_128",
    # "layers_512_256_128_residual_0_64_32",
    # "layers_512_256_256_residual_0_128_128",
}


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
        help="Network version which maps to certain network structure. See `PREDEFINED_NETWORKS` for the names",
    )
    args = parser.parse_args()

    if args.eval and args.trained_net_path == "":
        raise ValueError(
            "`--eval` is specified, so `--trained_net_path` must be specified"
        )

    return args


def make_state_from_grids(tile: Tile) -> Sequence[float]:
    return [float(value) for row in tile.grids for value in row]


# def make_state_one_hot(tile: Tile) -> Array:
#     def one_hot(v: int, size: int = 16) -> Sequence[float]:
#         loc = int(math.log2(float(v))) if v > 0 else 0
#         return [1.0 if i == loc else 0.0 for i in range(size)]

#     return [float(one_hot(value)) for row in tile.grids for value in row]


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


def load_net(network_version: str, in_features: int, out_features: int) -> nn.Module:
    hidden_layers: Tuple[int, ...]
    # residual_mid_feature_sizes: List[int]
    if network_version == "layers_1024_512_256":
        hidden_layers = (1024, 512, 256)
        # residual_mid_feature_sizes = []
    # elif network_version == "layers_512_512_residual_0_128":
    #     hidden_layers = (512, 512)
    #     residual_mid_feature_sizes = [0, 128]
    # elif network_version == "layers_512_256_128_residual_0_64_32":
    #     hidden_layers = (512, 256, 128)
    #     residual_mid_feature_sizes = [0, 64, 32]
    # elif network_version == "layers_512_256_256_residual_0_128_128":
    #     hidden_layers = (512, 256, 256)
    #     residual_mid_feature_sizes = [0, 128, 128]
    else:
        raise NameError(
            f"Network version {network_version} not in {PREDEFINED_NETWORKS}."
        )

    policy_net: nn.Module = Net(hidden_layers, out_features, nn.relu)
    return policy_net


def train(
    show_board: bool,
    print_results: bool,
    output_json_prefix: str,
    output_net_prefix: str,
    max_iters: int,
    network_version: str,
    pre_trained_net_path: str = "",
):
    tile: Tile = Tile(width=4, height=4)
    plot_properties: PlotProperties = PlotProperties(fps=60, delay_after_plot=50)
    plotter: TilePlotter = TilePlotter(tile, plot_properties)
    game_engine: GameEngine = GameEngine(tile)

    move_failures: List[int] = []
    total_scores: List[int] = []
    max_grids: List[int] = []

    # DQN part
    in_features: int = tile.width * tile.height * 16
    out_features: int = len(Action)
    policy_net = load_net(
        network_version,
        in_features,
        out_features,
    )

    training_params = TrainingParameters(
        memory_capacity=20000,
        gamma=0.99,
        batch_size=128,
        lr=5e-3,
        lr_decay_milestones=[1500, 3000, 5000, 7000],
        lr_gamma=0.1,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1500,
        TAU=0.005,
        save_network_steps=200,
        print_loss_steps=50,
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

    dqn = DQN(in_features, policy_net, output_net_dir, training_params, rng)
    if pre_trained_net_path != "":
        dqn.load_model(pre_trained_net_path)

    iter = 0
    start_time = time.time()
    cur_state: Sequence[float] = flat_one_hot(tile.flattened(), 16)
    next_state: Sequence[float] = cur_state
    total_rewards: List[float] = []
    suc_move_statistics: Dict[Action, int] = defaultdict(int)
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

            # move_result = game_engine.move(action)

            if move_result.suc:
                suc_move_statistics[action] += 1
                # print(tile)
                game_engine.generate_new()
                reward += move_result.score
            else:
                move_failure += 1
                reward += INVALID_MOVEMENT_REWARD

            if game_engine.game_is_over:
                reward += GAME_OVER_REWARD  # + tile.max_grid())

            # Normalize reward by max grid
            # reward /= tile.max_grid()

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

            if game_engine.game_is_over:
                total_rewards.append(total_reward)
                total_reward = 0.0

            # dqn.push_transition_and_optimize_automatically(transition, output_net_dir)
            dqn.push_transition(transition)
            new_collect_count += 1
            if new_collect_count >= training_params.batch_size * 50:
                for _ in range(50):
                    _loss = dqn.optimize_model()
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
                move_failure = 0
                max_grids.append(tile.max_grid())
                total_scores.append(game_engine.score)
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
                game_engine.reset()

    write_json(move_failures, total_scores, max_grids, total_rewards, output_json_fn)
    elapsed_sec = time.time() - start_time
    print(
        f"Done running {max_iters} times of experiments in {round(elapsed_sec * 1000.0)} millisecond(s)."
    )
    print(f"See results in {output_json_fn}.")


# def eval_dqn(
#     show_board: bool,
#     print_results: bool,
#     output_json_prefix: str,
#     max_iters: int,
#     trained_net_path: str,
#     network_version: str,
# ):
#     tile: Tile = Tile(width=4, height=4)
#     plot_properties: PlotProperties = PlotProperties(fps=60, delay_after_plot=50)
#     plotter: TilePlotter = TilePlotter(tile, plot_properties)
#     game_engine: GameEngine = GameEngine(tile)

#     move_failures: List[int] = []
#     total_scores: List[int] = []
#     max_grids: List[int] = []

#     # DQN part
#     in_features: int = tile.width * tile.height * 16
#     out_features: int = len(Action)
#     policy_net = load_net(network_version, in_features, out_features).policy_net

#     policy_net.eval()
#     try:
#         state_dict = torch.load(trained_net_path)
#     except FileNotFoundError:
#         try:
#             print(f"Loading model from {trained_net_path} failed.")
#             if "rl_2048/" in trained_net_path:
#                 search_in_parent_path: str = trained_net_path.replace("rl_2048/", "")
#                 print(f"Try to load model from {search_in_parent_path}")
#                 state_dict = torch.load(search_in_parent_path)
#             else:
#                 raise
#         except FileNotFoundError:
#             print(f"Loading model from {search_in_parent_path} still failed.")
#             raise

#     policy_net.load_state_dict(state_dict)

#     move_failure = 0
#     date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_json_fn = f"{output_json_prefix}_{date_time_str}.json"

#     iter = 0
#     start_time = time.time()
#     cur_state: Sequence[float] = make_state_one_hot(tile)

#     action_candidates: List[Action] = []
#     inf_times = []

#     prev_score: int = game_engine.score
#     score_not_increasing_count: int = 0
#     while max_iters < 0 or iter < max_iters:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 exit()

#             if event.type == pygame.KEYUP and event.key == pygame.K_r:
#                 game_engine.reset()
#                 prev_score = game_engine.score

#         if not game_engine.game_is_over:
#             start_inf_time = time.time()
#             policy_net_output = DQN.infer_action(policy_net, cur_state)
#             inf_times.append(time.time() - start_inf_time)
#             action: Action = policy_net_output.action

#             move_result: MoveResult = game_engine.move(action)
#             if move_result.suc:
#                 game_engine.generate_new()
#             else:
#                 move_failure += 1
#                 # action_candidates = [
#                 #     candidate for candidate in Action if candidate != action
#                 # ]
#                 # shuffle(action_candidates)
#                 # for action in action_candidates:
#                 #     move_result = game_engine.move(action)
#                 #     if move_result.suc:
#                 #         break
#                 #     move_failure += 1
#                 # if not move_result.suc:
#                 #     raise ValueError("Game is not over yet but all actions failed")

#             if game_engine.score == prev_score:
#                 score_not_increasing_count += 1
#             else:
#                 score_not_increasing_count = 0
#             prev_score = game_engine.score

#             cur_state = make_state_one_hot(tile)

#             if show_board:
#                 plotter.plot(game_engine.score)
#             else:
#                 tile.reset_animation_grids()

#             if game_engine.game_is_over:
#                 iter += 1
#                 if show_board:
#                     plotter.plot_game_over()
#                 move_failures.append(move_failure)
#                 move_failure = 0
#                 max_grids.append(tile.max_grid())
#                 total_scores.append(game_engine.score)
#                 if print_results:
#                     print(f"Move failures: {move_failures}")
#                     print(f"Total scores: {total_scores}")
#                     print(f"Max grids: {max_grids}")

#                 if iter % 10 == 0:
#                     write_json(
#                         move_failures,
#                         total_scores,
#                         max_grids,
#                         [],  # total_rewards,
#                         output_json_fn,
#                     )
#                 game_engine.reset()

#     write_json(move_failures, total_scores, max_grids, [], output_json_fn)
#     elapsed_sec = time.time() - start_time
#     print(
#         f"Done running {max_iters} times of experiments in {round(elapsed_sec * 1000.0)} millisecond(s)."
#     )
#     print(
#         f"Average inference time: {(sum(inf_times) / len(inf_times)) * 1000.0} millisecond(s)"
#     )
#     print(f"See results in {output_json_fn}.")


def main():
    args = parse_args()
    if args.eval:
        # eval_dqn(
        #     args.show_board,
        #     args.print_results,
        #     args.output_json_prefix,
        #     args.max_iters,
        #     args.trained_net_path,
        #     args.network_version,
        # )
        pass
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
