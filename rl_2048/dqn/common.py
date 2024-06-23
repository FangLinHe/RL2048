from collections.abc import Sequence
from enum import Enum
from typing import Any, NamedTuple, Union

PREDEFINED_NETWORKS: set[str] = {
    "layers_1024_512_256",
    "layers_512_512_residual_0_128",
    "layers_512_256_128_residual_0_64_32",
    "layers_512_256_256_residual_0_128_128",
}


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class DQNParameters(NamedTuple):
    memory_capacity: int = 1024
    batch_size: int = 64

    # for epsilon-greedy algorithm
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 400


class TrainingParameters(NamedTuple):
    gamma: float = 0.99
    optimizer: str = "adamw"
    lr: float = 0.001
    lr_decay_milestones: Union[int, list[int]] = 100
    lr_gamma: Union[float, list[float]] = 0.1
    loss_fn: str = "huber_loss"

    # update rate of the target network
    TAU: float = 0.005

    save_network_steps: int = 1000
    print_loss_steps: int = 100
    tb_write_steps: int = 50

    pretrained_net_path: str = ""


class Batch(NamedTuple):
    states: Sequence[Sequence[float]]
    actions: Sequence[int]
    next_states: Sequence[Sequence[float]]
    rewards: Sequence[float]
    games_over: Sequence[bool]


class PolicyNetOutput(NamedTuple):
    expected_value: float
    action: Action


Metrics = dict[str, Any]
