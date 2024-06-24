import random
from collections.abc import Sequence
from typing import NamedTuple

from rl_2048.dqn.common import Action, Batch


# N: number of grids in the board
class Transition(NamedTuple):
    state: Sequence[
        float
    ]  # size: N (if represented by float) or N*16 (if represented by one-hot)
    action: Action
    next_state: Sequence[float]  # not really used in training
    reward: float
    game_over: bool


class ReplayMemory:
    def __init__(self, capacity: int = 1024):
        self.capacity: int = capacity

        self.states: list[Sequence[float]] = [[] for _ in range(capacity)]
        self.actions: list[int] = [0 for _ in range(capacity)]
        self.next_states: list[Sequence[float]] = [[] for _ in range(capacity)]
        self.rewards: list[float] = [0.0 for _ in range(capacity)]
        self.games_over: list[bool] = [False for _ in range(capacity)]
        self.next_index: int = 0
        self.is_full: bool = False

    def reset(self):
        self.states = [[]] * self.capacity
        self.actions = [0] * self.capacity
        self.next_states = [[]] * self.capacity
        self.rewards = [0.0] * self.capacity
        self.games_over = [False] * self.capacity
        self.next_index = 0
        self.is_full = False

    def push(self, transition: Transition):
        self.states[self.next_index] = transition.state
        self.actions[self.next_index] = transition.action.value
        self.next_states[self.next_index] = transition.next_state
        self.rewards[self.next_index] = transition.reward
        self.games_over[self.next_index] = transition.game_over
        self.next_index += 1
        if self.next_index >= self.capacity:
            self.is_full = True
            self.next_index = 0

    def sample(self, batch_size: int) -> Batch:
        random_indices = random.sample(list(range(len(self))), batch_size)
        batch = Batch(
            [self.states[i] for i in random_indices],
            [self.actions[i] for i in random_indices],
            [self.next_states[i] for i in random_indices],
            [self.rewards[i] for i in random_indices],
            [self.games_over[i] for i in random_indices],
        )

        return batch

    def __len__(self):
        return self.capacity if self.is_full else self.next_index
