import random

from enum import Enum
from typing import List, NamedTuple, Sequence
import torch


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# N: number of grids in the board
class Transition(NamedTuple):
    state: Sequence[float]  # size: N (if represented by float) or N*16 (if represented by one-hot)
    action: Action
    next_state: Sequence[float]  # not really used in training
    reward: float
    game_over: bool

class Batch(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    games_over: torch.Tensor

class ReplayMemory:
    def __init__(self, capacity: int = 1024):
        self.states: List[Sequence[float]] = []
        self.actions: List[int] = []
        self.next_states: List[Sequence[float]] = []
        self.rewards: List[float] = []
        self.games_over: List[bool] = []

        self.capacity: int = capacity
    
    def is_full(self) -> bool:
        return len(self) >= self.capacity

    def reset(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.games_over = []


    def push(self, transition: Transition) -> bool:
        # Simply not appending if buffer is full
        if self.is_full():
            return False
        self.states.append(transition.state)
        self.actions.append(transition.action.value)
        self.next_states.append(transition.next_state)
        self.rewards.append(transition.reward)
        self.games_over.append(transition.game_over)

        return True

    def sample(self, batch_size: int) -> Batch:
        random_indices = random.sample(list(range(len(self))), batch_size)
        return Batch(
            torch.tensor([self.states[i] for i in random_indices]),
            torch.tensor([self.actions[i] for i in random_indices], dtype=torch.int64).view((-1, 1)),
            torch.tensor([self.next_states[i] for i in random_indices]),
            torch.tensor([self.rewards[i] for i in random_indices]).view((-1, 1)),
            torch.tensor([self.games_over[i] for i in random_indices], dtype=torch.bool).view((-1, 1))
        )

    def __len__(self):
        return len(self.states)

if __name__ == "__main__":
    memory = ReplayMemory()
    t1 = Transition(
        state=[1.0, 0.5],
        action=Action.UP,
        next_state=[2.0, 0.0],
        reward=10.0,
        game_over=False
    )
    t2 = Transition(
        state=[2.0, 0.0],
        action=Action.LEFT,
        next_state=[-0.5, 1.0],
        reward=-1.0,
        game_over=False
    )
    memory.push(t1)
    memory.push(t2)
    print(memory.sample(2))