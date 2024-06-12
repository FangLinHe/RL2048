from enum import Enum
from typing import NamedTuple, Union

from jaxtyping import Array


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class TrainingParameters(NamedTuple):
    memory_capacity: int = 1024
    gamma: float = 0.99
    batch_size: int = 64
    optimizer: str = "adamw"
    lr: float = 0.001
    lr_decay_milestones: Union[int, list[int]] = 100
    lr_gamma: Union[float, list[float]] = 0.1
    loss_fn: str = "huber_loss"

    # for epsilon-greedy algorithm
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 400

    # update rate of the target network
    TAU: float = 0.005

    save_network_steps: int = 1000
    print_loss_steps: int = 100
    tb_write_steps: int = 50

    pretrained_net_path: str = ""


class Batch(NamedTuple):
    states: Array
    actions: Array
    next_states: Array
    rewards: Array
    games_over: Array


class PolicyNetOutput(NamedTuple):
    expected_value: float
    action: Action
