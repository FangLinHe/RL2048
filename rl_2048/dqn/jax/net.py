import functools
from collections.abc import Mapping
from typing import Any, Callable, NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jax import Array
from typing_extensions import TypeAlias

from rl_2048.dqn.jax.replay_memory import Batch

Params: TypeAlias = FrozenDict[str, Any]
Variables: TypeAlias = Union[FrozenDict[str, Mapping[str, Any]], dict[str, Any]]
GradFn: TypeAlias = Callable[
    [Params],
    tuple[Array, Array],
]


PREDEFINED_NETWORKS: set[str] = {
    "layers_1024_512_256",
    "layers_512_512_residual_0_128",
    "layers_512_256_128_residual_0_64_32",
    "layers_512_256_256_residual_0_128_128",
}


class BNTrainState(TrainState):
    """TrainState that is used for modules with BatchNorm layers."""

    batch_stats: Any


class ResidualBlock(nn.Module):
    in_dim: int
    mid_dim: int
    out_dim: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, x: Array, train: bool):
        residual: Array = x
        x = nn.Dense(self.mid_dim, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.mid_dim, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.out_dim, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        if residual.shape != x.shape:
            pool_size: int = self.in_dim // self.out_dim
            residual = nn.avg_pool(
                residual[:, :, None],
                window_shape=(
                    1,
                    pool_size,
                ),
                strides=(
                    1,
                    pool_size,
                ),
            )[:, :, 0]

        return x + residual


# All Flax Modules are Python 3.7 dataclasses.
# Since dataclasses take over __init__, you should instead override setup(),
# which is automatically called to initialize the module.
class Net(nn.Module):
    hidden_dims: tuple[int, ...]
    output_dim: int
    net_activation_fn: Callable
    residual_mid_dims: tuple[int, ...]

    def check_correctness(self):
        N_hidden, N_res = len(self.hidden_dims), len(self.residual_mid_dims)
        if N_hidden != N_res:
            if N_res == 0:
                self.residual_mid_dims = tuple(0 for _ in range(N_hidden))
            else:
                raise ValueError(
                    "`residual_mid_dims` should be either empty or have the same "
                    f"length as `hidden_dims` ({N_hidden}), but got ({N_res})"
                )

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        in_dim: int = x.shape[-1]
        for residual_mid_dim, hidden_dim in zip(
            self.residual_mid_dims, self.hidden_dims
        ):
            if residual_mid_dim == 0:
                x = nn.Dense(features=hidden_dim, use_bias=False)(x)
                x = nn.BatchNorm(use_running_average=not train)(x)
            else:
                x = ResidualBlock(
                    in_dim, residual_mid_dim, hidden_dim, self.net_activation_fn
                )(x, train)
            in_dim = hidden_dim
            x = self.net_activation_fn(x)

        return nn.Dense(features=self.output_dim)(x)


def create_train_state(
    rng: Array,
    net: nn.Module,
    input_dim: int,
    optimizer: str,
    lr_scheduler_fn: optax.Schedule,
) -> BNTrainState:
    variables: Variables = net.init(rng, jnp.ones([2, input_dim]))
    optimizer_fn = getattr(optax, optimizer)
    tx: optax.GradientTransformation = optimizer_fn(lr_scheduler_fn)
    return BNTrainState.create(
        apply_fn=net.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=tx,
    )


class JaxBatch(NamedTuple):
    states: Array
    actions: Array
    next_states: Array
    rewards: Array
    games_over: Array


def to_jax_batch(batch: Batch) -> JaxBatch:
    return JaxBatch(
        states=jnp.array(np.array(batch.states)),
        actions=jnp.array(np.array(batch.actions)).reshape((-1, 1)),
        next_states=jnp.array(np.array(batch.next_states)),
        rewards=jnp.array(np.array(batch.rewards)).reshape((-1, 1)),
        games_over=jnp.array(np.array(batch.games_over)).reshape((-1, 1)),
    )


@functools.partial(jax.jit, static_argnums=(3, 4))
def train_step(
    train_state: BNTrainState,
    input_batch: JaxBatch,
    targets: Array,
    learning_rate_fn: optax.Schedule,
    optax_loss_fn: Callable,
) -> tuple[BNTrainState, Any, float]:
    """Computes gradients and loss for a single batch."""

    def loss_fn(params) -> tuple[Array, tuple[Array, Array]]:
        raw_pred, updates = train_state.apply_fn(
            {"params": params, "batch_stats": train_state.batch_stats},
            x=input_batch.states,
            train=True,
            mutable=["batch_stats"],
        )

        predictions: Array = jnp.take_along_axis(raw_pred, input_batch.actions, axis=1)
        loss = jnp.mean(optax_loss_fn(predictions, targets))
        return loss, (predictions, updates)

    grad_fn: GradFn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (_predictions, updates)), grads = grad_fn(train_state.params)

    lr = learning_rate_fn(train_state.step)

    train_state = train_state.apply_gradients(grads=grads)
    train_state = train_state.replace(batch_stats=updates["batch_stats"])

    return train_state, loss, lr


@jax.jit
def eval_forward(train_state: BNTrainState, inputs: Array) -> Array:
    predictions = train_state.apply_fn(
        {"params": train_state.params, "batch_stats": train_state.batch_stats},
        x=inputs,
        train=False,
    )
    return predictions


@jax.jit
def train_forward(train_state: BNTrainState, inputs: Array) -> tuple[Array, Variables]:
    predictions, updates = train_state.apply_fn(
        {"params": train_state.params, "batch_stats": train_state.batch_stats},
        x=inputs,
        train=True,
        mutable=["batch_stats"],
    )
    return predictions, updates


def load_predefined_net(network_version: str, out_features: int) -> Net:
    if network_version not in PREDEFINED_NETWORKS:
        raise NameError(
            f"Network version {network_version} not in {PREDEFINED_NETWORKS}."
        )

    hidden_layers: tuple[int, ...]
    residual_mid_feature_sizes: tuple[int, ...]
    if network_version == "layers_1024_512_256":
        hidden_layers = (1024, 512, 256)
        residual_mid_feature_sizes = ()
    elif network_version == "layers_512_512_residual_0_128":
        hidden_layers = (512, 512)
        residual_mid_feature_sizes = (0, 128)
    elif network_version == "layers_512_256_128_residual_0_64_32":
        hidden_layers = (512, 256, 128)
        residual_mid_feature_sizes = (0, 64, 32)
    elif network_version == "layers_512_256_256_residual_0_128_128":
        hidden_layers = (512, 256, 256)
        residual_mid_feature_sizes = (0, 128, 128)

    policy_net: Net = Net(
        hidden_dims=hidden_layers,
        output_dim=out_features,
        net_activation_fn=nn.relu,
        residual_mid_dims=residual_mid_feature_sizes,
    )
    return policy_net
