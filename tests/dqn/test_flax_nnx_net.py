import tempfile

import jax.numpy as jnp
import pytest
from flax import nnx
from jax import Array
from jax import random as jrandom

from rl_2048.dqn import DQN
from rl_2048.dqn.common import (
    PREDEFINED_NETWORKS,
    Action,
    DQNParameters,
    TrainingParameters,
)
from rl_2048.dqn.flax_nnx_net import (
    FlaxNnxPolicyNet,
    Net,
    ResidualBlock,
    _load_predefined_net,
)
from rl_2048.dqn.replay_memory import Transition


@pytest.fixture
def rngs() -> nnx.Rngs:
    return nnx.Rngs(params=0)


@pytest.fixture
def input_array(batch: int = 1, dim: int = 4) -> Array:
    return jnp.ones((batch, dim))


@pytest.fixture
def output_dim() -> int:
    return 4


def test_residual_block(rngs: nnx.Rngs, input_array: Array):
    for mid in (2, 4, 6):
        ResidualBlock(4, mid, 4, nnx.relu, rngs)(input_array)
        ResidualBlock(4, mid, 2, nnx.relu, rngs)(input_array)

    # out_dim must be smaller or equal to in_dim
    for mid in (2, 4, 6):
        with pytest.raises(TypeError):
            ResidualBlock(4, mid, 8, nnx.relu, rngs)(input_array)


def test_predefined_nets(rngs: nnx.Rngs, input_array: Array, output_dim: int):
    rngs = nnx.Rngs(params=0)

    for network_version in PREDEFINED_NETWORKS:
        _load_predefined_net(network_version, input_array.shape[1], output_dim, rngs)(
            input_array
        )


def test_invalid_nets(rngs: nnx.Rngs, input_array: Array, output_dim: int):
    input_dim: int = input_array.shape[1]

    with pytest.raises(ValueError):
        Net(
            input_dim,
            (2, 2, 2),
            output_dim,
            nnx.relu,
            (2, 2),
            rngs,
        )

    with pytest.raises(NameError):
        _load_predefined_net("foo", input_dim, output_dim, rngs)


def test_jax_policy_net(rngs: nnx.Rngs):
    input_dim = 100
    output_dim = 4
    dqn_params = DQNParameters(
        memory_capacity=4,
        batch_size=2,
        eps_start=0.0,
        eps_end=0.0,
    )
    training_params = TrainingParameters(
        gamma=0.99,
        lr=0.1,
    )
    t1 = Transition(
        state=jrandom.normal(rngs.params(), shape=(input_dim,)).tolist(),
        action=Action.UP,
        next_state=jrandom.normal(rngs.params(), shape=(input_dim,)).tolist(),
        reward=10.0,
        game_over=False,
    )
    t2 = Transition(
        state=jrandom.normal(rngs.params(), shape=(input_dim,)).tolist(),
        action=Action.LEFT,
        next_state=jrandom.normal(rngs.params(), shape=(input_dim,)).tolist(),
        reward=-1.0,
        game_over=False,
    )

    test_feature = jrandom.normal(rngs.params(), shape=(input_dim,)).tolist()

    for network_version in PREDEFINED_NETWORKS:
        policy_net = FlaxNnxPolicyNet(
            network_version, input_dim, output_dim, rngs, training_params
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dqn = DQN(policy_net, dqn_params)

            dqn.push_transition(t1)
            dqn.push_transition(t2)
            loss = dqn.optimize_model()
            assert loss != 0.0

            _ = dqn.get_action_epsilon_greedy(t2.state)

            model_path = dqn.save_model(tmp_dir)

            policy_net_2 = FlaxNnxPolicyNet(
                network_version, input_dim, output_dim, rngs
            )
            dqn_load_model = DQN(policy_net_2)
            assert dqn_load_model.predict(test_feature).expected_value != pytest.approx(
                dqn.predict(test_feature).expected_value
            )
            dqn_load_model.load_model(model_path)

            assert dqn_load_model.predict(test_feature).expected_value == pytest.approx(
                dqn.predict(test_feature).expected_value
            )
