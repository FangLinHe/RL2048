import tempfile

from flax import linen as nn
from jax import Array
from jax import random as jrandom

from rl_2048.jax_dqn.dqn import DQN, TrainingParameters
from rl_2048.jax_dqn.net import Net
from rl_2048.jax_dqn.replay_memory import Action, Transition


def test_dqn():
    input_dim = 2
    hidden_dims = (3,)
    output_dim = 4
    training_params = TrainingParameters(
        memory_capacity=2,
        gamma=0.99,
        batch_size=2,
        lr=0.001,
        sgd_momentum=0.9,
        eps_start=0.0,
        eps_end=0.0,
    )
    rng: Array = jrandom.key(0)
    policy_net: nn.Module = Net(hidden_dims, output_dim, nn.relu)

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

    with tempfile.TemporaryDirectory() as tmp_dir:
        dqn = DQN(input_dim, policy_net, tmp_dir, training_params, rng)

        dqn.push_transition(t1)
        dqn.push_transition(t2)
        loss = dqn.optimize_model()
        assert loss != 0.0

        print(dqn.get_action_epsilon_greedy(t2.state))

        model_path = dqn.save_model()
        dqn.load_model(model_path)