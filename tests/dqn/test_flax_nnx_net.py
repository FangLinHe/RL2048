import jax.numpy as jnp
import pytest
from flax import nnx
from jax import Array

from rl_2048.dqn.common import PREDEFINED_NETWORKS
from rl_2048.dqn.flax_nnx_net import Net, ResidualBlock, _load_predefined_net


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
