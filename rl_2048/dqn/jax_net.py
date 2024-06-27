import functools
import os
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.checkpoints import restore_checkpoint, save_checkpoint
from flax.training.train_state import TrainState
from jax import Array
from jax.tree_util import tree_map
from jaxtyping import PyTree
from typing_extensions import TypeAlias

from rl_2048.dqn.common import (
    PREDEFINED_NETWORKS,
    Action,
    Batch,
    Metrics,
    PolicyNetOutput,
    TrainingParameters,
)
from rl_2048.dqn.jax_utils import JaxBatch, _create_lr_scheduler, to_jax_batch
from rl_2048.dqn.protocols import PolicyNet

Params: TypeAlias = FrozenDict[str, Any]
Variables: TypeAlias = Union[FrozenDict[str, Mapping[str, Any]], dict[str, Any]]
GradFn: TypeAlias = Callable[
    [Params],
    tuple[Array, Array],
]


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


@functools.partial(jax.jit, static_argnums=(3, 4))
def train_step(
    train_state: BNTrainState,
    input_batch: JaxBatch,
    targets: Array,
    learning_rate_fn: optax.Schedule,
    optax_loss_fn: Callable,
) -> tuple[BNTrainState, Any, Union[int, Array], float]:
    """Computes gradients and loss for a single batch."""

    def loss_fn(params) -> tuple[Array, tuple[Array, Optional[Array]]]:
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

    return train_state, loss, train_state.step, lr


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


def _load_predefined_net(network_version: str, out_features: int) -> Net:
    if network_version not in PREDEFINED_NETWORKS:
        raise NameError(
            f"Network version {network_version} not in {PREDEFINED_NETWORKS}."
        )

    hidden_layers: tuple[int, ...]
    residual_mid_feature_sizes: tuple[int, ...]
    if network_version == "layers_1024_512_256":
        hidden_layers = (1024, 512, 256)
        residual_mid_feature_sizes = (0, 0, 0)
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


class TrainingElements:
    """Class for keeping track of training variables"""

    params: TrainingParameters
    loss_fn: Callable
    lr_scheduler: optax.Schedule
    policy_net_train_state: BNTrainState
    target_net_train_state: BNTrainState
    step_count: int

    def __init__(
        self,
        training_params: TrainingParameters,
        random_key: Array,
        policy_net: Net,
        in_features: int,
    ):
        self.params = training_params
        self.loss_fn = getattr(optax, training_params.loss_fn)
        self.lr_scheduler = _create_lr_scheduler(training_params)
        self.policy_net_train_state: BNTrainState = create_train_state(
            random_key,
            policy_net,
            in_features,
            training_params.optimizer,
            self.lr_scheduler,
        )
        self.target_net_train_state: BNTrainState = create_train_state(
            random_key,
            policy_net,
            in_features,
            training_params.optimizer,
            self.lr_scheduler,
        )
        self.step_count = 0


class JaxPolicyNet(PolicyNet):
    """
    Implements protocal `PolicyNet` with Jax (see rl_2048/dqn/protocols.py)
    """

    def __init__(
        self,
        network_version: str,
        in_features: int,
        out_features: int,
        random_key: Array,
        training_params: Optional[TrainingParameters] = None,
    ):
        self.policy_net: Net = _load_predefined_net(network_version, out_features)
        self.policy_net_apply: Callable = jax.jit(self.policy_net.apply)
        self.policy_net_variables: PyTree = {}

        self.random_key: Array = random_key
        self.training: Optional[TrainingElements]

        if training_params is None:
            self.training = None
        else:
            self.training = TrainingElements(
                training_params, random_key, self.policy_net, in_features
            )

    def check_correctness(self):
        self.policy_net.check_correctness()

    def predict(self, state_feature: Sequence[float]) -> PolicyNetOutput:
        state_array: Array = jnp.array(np.array(state_feature))[None, :]
        if self.training is None:
            raw_values: Array = self.policy_net_apply(
                self.policy_net_variables,
                state_array,
            )[0]
        else:
            net_train_states = self.training.policy_net_train_state
            net_params = {
                "params": net_train_states.params,
                "batch_stats": net_train_states.batch_stats,
            }

            raw_values = net_train_states.apply_fn(
                net_params,
                x=state_array,
                train=False,
            )[0]

        best_action: int = jnp.argmax(raw_values).item()
        best_value: float = raw_values[best_action].item()
        return PolicyNetOutput(best_value, Action(best_action))

    def error_msg(self) -> str:
        return (
            "TorchPolicyNet is not initailized with training_params. "
            "This function is not supported."
        )

    def optimize(self, batch: Batch) -> Metrics:
        if self.training is None:
            raise ValueError(self.error_msg())

        jax_batch: JaxBatch = to_jax_batch(batch)

        next_value_predictions = eval_forward(
            self.training.target_net_train_state, jax_batch.next_states
        )
        next_state_values = next_value_predictions.max(axis=1, keepdims=True)
        expected_state_action_values: Array = jax_batch.rewards + (
            self.training.params.gamma * next_state_values
        ) * (1.0 - jax_batch.games_over)
        self.training.policy_net_train_state, loss, step_state, lr = train_step(
            self.training.policy_net_train_state,
            jax_batch,
            expected_state_action_values,
            self.training.lr_scheduler,
            self.training.loss_fn,
        )
        loss_val: float = loss.item()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        tau: float = self.training.params.TAU
        target_net_params = tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            self.training.policy_net_train_state.params,
            self.training.target_net_train_state.params,
        )
        self.training.target_net_train_state = (
            self.training.target_net_train_state.replace(params=target_net_params)
        )
        if isinstance(
            self.training.policy_net_train_state, BNTrainState
        ) and isinstance(self.training.target_net_train_state, BNTrainState):
            target_net_batch_stats = tree_map(
                lambda p, tp: p * tau + tp * (1 - tau),
                self.training.policy_net_train_state.batch_stats,
                self.training.target_net_train_state.batch_stats,
            )
            self.training.target_net_train_state = (
                self.training.target_net_train_state.replace(
                    batch_stats=target_net_batch_stats
                )
            )

        self.training.step_count += 1

        step: int = step_state if isinstance(step_state, int) else step_state.item()

        return {"loss": loss_val, "step": step, "lr": lr}

    def save(self, model_path: str) -> str:
        if self.training is None:
            raise ValueError(self.error_msg())
        saved_path: str = save_checkpoint(
            ckpt_dir=os.path.abspath(model_path),
            target=self.training.policy_net_train_state,
            step=self.training.step_count,
            keep=10,
        )

        return saved_path

    def load(self, model_path: str):
        policy_net_variables = restore_checkpoint(model_path, None)
        self.policy_net_variables = {
            "params": policy_net_variables["params"],
            "batch_stats": policy_net_variables["batch_stats"],
        }

        if self.training is None:
            return

        self.training.policy_net_train_state = restore_checkpoint(
            ckpt_dir=os.path.dirname(model_path),
            target=self.training.policy_net_train_state,
        )
        # Reset step to 0, so LR scheduler works as expected
        self.training.policy_net_train_state = (
            self.training.policy_net_train_state.replace(step=0)
        )
        self.training.target_net_train_state = (
            self.training.target_net_train_state.replace(
                params=self.training.policy_net_train_state.params,
                batch_stats=self.training.policy_net_train_state.batch_stats,
            )
        )
