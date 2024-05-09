import random

from enum import Enum
from typing import List, NamedTuple, Optional, Sequence
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

class Batch(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor

class ReplayMemory:
    def __init__(self, capacity: int = 1024):
        self.states: List[Sequence[float]] = []
        self.actions: List[int] = []
        self.next_states: List[Sequence[float]] = []
        self.rewards: List[float] = []

        self.capacity: int = capacity
    
    def reset(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []


    def push(self, transition: Transition) -> bool:
        # Simply not appending if buffer is full
        if len(self) >= self.capacity:
            return False
        self.states.append(transition.state)
        self.actions.append(transition.action.value)
        self.next_states.append(transition.next_state)
        self.rewards.append(transition.reward)

        return True

    def sample(self, batch_size: int) -> Batch:
        return Batch(
            torch.tensor(random.sample(self.states, batch_size)),
            torch.tensor(random.sample(self.actions, batch_size), dtype=torch.int64).view((-1, 1)),
            torch.tensor(random.sample(self.next_states, batch_size)),
            torch.tensor(random.sample(self.rewards, batch_size)).view((-1, 1))
        )

    def __len__(self):
        return len(self.states)

if __name__ == "__main__":
    memory = ReplayMemory()
    t1 = Transition(
        state=[1.0, 0.5],
        action=Action.UP,
        next_state=[2.0, 0.0],
        reward=10.0
    )
    t2 = Transition(
        state=[2.0, 0.0],
        action=Action.LEFT,
        next_state=[-0.5, 1.0],
        reward=-1.0
    )
    memory.push(t1)
    memory.push(t2)
    print(memory.sample(2))
