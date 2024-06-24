import jax.numpy as jnp
import pytest
from flax import nnx

from rl_2048.dqn.flax_nnx_net import ResidualBlock


def test_residual_block():
    rngs = nnx.Rngs(params=0)
    x = jnp.ones((1, 4))
    for mid in (2, 4, 6):
        ResidualBlock(4, mid, 4, nnx.relu, rngs)(x)
        ResidualBlock(4, mid, 2, nnx.relu, rngs)(x)

    # out_dim must be smaller or equal to in_dim
    for mid in (2, 4, 6):
        with pytest.raises(TypeError):
            ResidualBlock(4, mid, 8, nnx.relu, rngs)(x)
