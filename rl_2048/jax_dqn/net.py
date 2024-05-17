from typing import Any, Callable, Mapping, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jax import Array
from typing_extensions import TypeAlias

from rl_2048.jax_dqn.replay_memory import Batch

Params: TypeAlias = FrozenDict[str, Any]
Variables: TypeAlias = Union[FrozenDict[str, Mapping[str, Any]], dict[str, Any]]
GradFn: TypeAlias = Callable[
    [Params],
    Tuple[Array, Array],
]


class BNTrainState(TrainState):
    """TrainState that is used for modules with BatchNorm layers."""

    batch_stats: Any


# All Flax Modules are Python 3.7 dataclasses.
# Since dataclasses take over __init__, you should instead override setup(),
# which is automatically called to initialize the module.
class Net(nn.Module):
    hidden_dims: Tuple[int, ...]
    output_dim: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(features=hidden_dim, use_bias=False)(x)
            # https://flax.readthedocs.io/en/latest/guides/training_techniques/batch_norm.html
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = self.activation_fn(x)

        return nn.Dense(features=self.output_dim)(x)


def create_train_state(
    rng: Array,
    net: nn.Module,
    input_dim: int,
    learning_rate: float,
    momentum: float = 0.9,
) -> BNTrainState:
    variables: Variables = net.init(rng, jnp.ones([1, input_dim]))
    tx: optax.GradientTransformation = optax.sgd(learning_rate, momentum)
    return BNTrainState.create(
        apply_fn=net.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=tx,
    )


@jax.jit
def train_step(
    train_state: BNTrainState, input_batch: Batch, targets: Array
) -> Tuple[BNTrainState, Any]:
    """Computes gradients and loss for a single batch."""

    def loss_fn(params) -> Tuple[Array, Tuple[Array, Array]]:
        raw_pred, updates = train_state.apply_fn(
            {"params": params, "batch_stats": train_state.batch_stats},
            x=input_batch.states,
            train=True,
            mutable=["batch_stats"],
        )

        predictions: Array = jnp.take_along_axis(raw_pred, input_batch.actions, axis=1)
        loss = jnp.mean(optax.squared_error(predictions, targets))
        return loss, (predictions, updates)

    grad_fn: GradFn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (_predictions, updates)), grads = grad_fn(train_state.params)

    train_state = train_state.apply_gradients(grads=grads)
    train_state = train_state.replace(batch_stats=updates["batch_stats"])

    return train_state, loss


@jax.jit
def eval_forward(train_state: BNTrainState, inputs: Array) -> Array:
    predictions = train_state.apply_fn(
        {"params": train_state.params, "batch_stats": train_state.batch_stats},
        x=inputs,
        train=False,
    )
    return predictions


@jax.jit
def train_forward(train_state: BNTrainState, inputs: Array) -> Tuple[Array, Variables]:
    predictions, updates = train_state.apply_fn(
        {"params": train_state.params, "batch_stats": train_state.batch_stats},
        x=inputs,
        train=True,
        mutable=["batch_stats"],
    )
    return predictions, updates
