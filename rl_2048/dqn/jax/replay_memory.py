import random
from collections.abc import Sequence
from typing import NamedTuple

import jax.random as jrandom
from jax import Array

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
    def __init__(self, rng: Array, capacity: int = 1024):
        self.rng = rng
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
        # batch = Batch(
        #     jnp.array(np.array([self.states[i] for i in random_indices])),
        #     jnp.array(
        #         np.array([self.actions[i] for i in random_indices]).reshape((-1, 1))
        #     ),
        #     jnp.array(
        #         np.array([self.next_states[i] for i in random_indices]),
        #     ),
        #     jnp.array(
        #         np.array([self.rewards[i] for i in random_indices]).reshape((-1, 1))
        #     ),
        #     jnp.array(
        #         np.array([float(self.games_over[i]) for i in random_indices]).reshape(
        #             (-1, 1)
        #         )
        #     ),
        # )
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


if __name__ == "__main__":
    rng = jrandom.key(0)
    memory = ReplayMemory(rng)
    t1 = Transition(
        state=[1.0, 0.5],
        action=Action.UP,
        next_state=[2.0, 0.0],
        reward=10.0,
        game_over=False,
    )
    t2 = Transition(
        state=[2.0, 0.0],
        action=Action.LEFT,
        next_state=[-0.5, 1.0],
        reward=-1.0,
        game_over=False,
    )
    memory.push(t1)
    memory.push(t2)
    print(memory.sample(2))
