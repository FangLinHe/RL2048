import jax.random as jrandom
from jax import Array

from rl_2048.dqn.common import Action
from rl_2048.dqn.jax.net import JaxBatch, to_jax_batch
from rl_2048.dqn.jax.replay_memory import ReplayMemory, Transition

all_memory_fields = {"states", "actions", "next_states", "rewards", "games_over"}


def check_size(memory: ReplayMemory, expected_size: int):
    assert memory.is_full == (expected_size == memory.capacity)
    assert len(memory) == expected_size


def test_replay_memory():
    rng: Array = jrandom.key(0)
    capacity: int = 4
    state_size: int = 2
    memory = ReplayMemory(rng, capacity)
    t1 = Transition(
        state=[1.0] + [0.0 for _ in range(state_size - 1)],
        action=Action.UP,
        next_state=[2.0] + [0.0 for _ in range(state_size - 1)],
        reward=2.0,
        game_over=False,
    )
    t2 = Transition(
        state=[2.0] + [0.0 for _ in range(state_size - 1)],
        action=Action.DOWN,
        next_state=[3.0] + [0.0 for _ in range(state_size - 1)],
        reward=3.0,
        game_over=False,
    )
    t3 = Transition(
        state=[3.0] + [0.0 for _ in range(state_size - 1)],
        action=Action.LEFT,
        next_state=[4.0] + [0.0 for _ in range(state_size - 1)],
        reward=4.0,
        game_over=False,
    )
    t4 = Transition(
        state=[4.0] + [0.0 for _ in range(state_size - 1)],
        action=Action.RIGHT,
        next_state=[0.0] + [0.0 for _ in range(state_size - 1)],
        reward=-1.0,
        game_over=True,
    )

    assert memory.capacity == capacity
    check_size(memory, 0)

    assert memory.next_index == 0
    memory.push(t1)
    check_size(memory, 1)
    assert memory.next_index == 1
    memory.push(t2)
    check_size(memory, 2)
    assert memory.next_index == 2
    memory.push(t3)
    check_size(memory, 3)
    assert memory.next_index == 3
    memory.push(t4)
    check_size(memory, 4)
    assert memory.next_index == 0

    sample_size: int = 3
    batch: JaxBatch = to_jax_batch(memory.sample(sample_size))
    expected_shapes = {
        f: (sample_size, (state_size if "states" in f else 1)) for f in batch._fields
    }
    assert all(getattr(batch, f).shape == expected_shapes[f] for f in batch._fields)
