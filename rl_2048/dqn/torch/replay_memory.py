import random
from collections import deque
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

        self.states: deque[Sequence[float]] = deque(maxlen=self.capacity)
        self.actions: deque[int] = deque(maxlen=self.capacity)
        self.next_states: deque[Sequence[float]] = deque(maxlen=self.capacity)
        self.rewards: deque[float] = deque(maxlen=self.capacity)
        self.games_over: deque[bool] = deque(maxlen=self.capacity)

    def is_full(self) -> bool:
        return len(self) >= self.capacity

    def reset(self):
        self.states = deque(maxlen=self.capacity)
        self.actions = deque(maxlen=self.capacity)
        self.next_states = deque(maxlen=self.capacity)
        self.rewards = deque(maxlen=self.capacity)
        self.games_over = deque(maxlen=self.capacity)

    def push(self, transition: Transition):
        self.states.append(transition.state)
        self.actions.append(transition.action.value)
        self.next_states.append(transition.next_state)
        self.rewards.append(transition.reward)
        self.games_over.append(transition.game_over)

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
        return len(self.states)


if __name__ == "__main__":
    memory = ReplayMemory()
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
