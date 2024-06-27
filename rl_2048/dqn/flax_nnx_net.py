"""
Implement the protocol `PolicyNet` with flax.nnx
"""

import functools
import os
from collections.abc import Sequence
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax.training.checkpoints import restore_checkpoint, save_checkpoint
from jax.tree_util import tree_map
from jaxtyping import Array

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


class ResidualBlock(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
        activation_fn: Callable,
        rngs: nnx.Rngs,
    ):
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.activation_fn = activation_fn

        self.linear1 = nnx.Linear(in_dim, mid_dim, use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(mid_dim, rngs=rngs)
        self.linear2 = nnx.Linear(mid_dim, mid_dim, use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(mid_dim, rngs=rngs)
        self.linear3 = nnx.Linear(mid_dim, out_dim, use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(out_dim, rngs=rngs)

    def __call__(self, x: Array):
        residual: Array = x
        x = self.bn1(self.linear1(x))
        x = self.activation_fn(x)
        x = self.activation_fn(self.bn2(self.linear2(x)))
        x = self.bn3(self.linear3(x))

        if residual.shape != x.shape:
            pool_size: int = self.in_dim // self.out_dim
            print(residual.shape)
            residual = nnx.avg_pool(
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


class Net(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        net_activation_fn: Callable,
        residual_mid_dims: tuple[int, ...],
        rngs: nnx.Rngs,
    ):
        if len(residual_mid_dims) == 0:
            residual_mid_dims = tuple(0 for _ in range(len(hidden_dims)))

        def validate_args():
            N_hidden, N_res = len(hidden_dims), len(residual_mid_dims)
            if N_hidden != N_res:
                raise ValueError(
                    "`residual_mid_dims` should be either empty or have the same "
                    f"length as `hidden_dims` ({N_hidden}), but got ({N_res})"
                )

        validate_args()

        layers: list[Callable] = []
        for residual_mid_dim, hidden_dim in zip(residual_mid_dims, hidden_dims):
            block: list[Callable] = []
            if residual_mid_dim == 0:
                block.append(nnx.Linear(in_dim, hidden_dim, use_bias=False, rngs=rngs))
                block.append(nnx.BatchNorm(hidden_dim, rngs=rngs))
            else:
                block.append(
                    ResidualBlock(
                        in_dim, residual_mid_dim, hidden_dim, net_activation_fn, rngs
                    )
                )
            in_dim = hidden_dim
            block.append(net_activation_fn)
            layers.append(nnx.Sequential(*block))

        layers.append(nnx.Linear(in_dim, output_dim, rngs=rngs))

        self.layers = nnx.Sequential(*layers)

    def __call__(self, x: Array):
        return self.layers(x)


def _load_predefined_net(
    network_version: str, in_dim: int, output_dim: int, rngs: nnx.Rngs
) -> Net:
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
        in_dim,
        hidden_layers,
        output_dim,
        nnx.relu,
        residual_mid_feature_sizes,
        rngs,
    )
    return policy_net


class TrainingElements:
    """Class for keeping track of training variables"""

    def __init__(
        self,
        training_params: TrainingParameters,
        policy_net: Net,
        target_net: Net,
    ):
        self.target_net: Net = target_net
        self.target_net.eval()
        self.params: TrainingParameters = training_params
        self.loss_fn: Callable = getattr(optax, training_params.loss_fn)

        self.lr_scheduler: optax.Schedule = _create_lr_scheduler(training_params)
        optimizer_fn: Callable = getattr(optax, training_params.optimizer)
        tx: optax.GradientTransformation = optimizer_fn(self.lr_scheduler)
        self.state = nnx.Optimizer(policy_net, tx)


@functools.partial(nnx.jit, static_argnums=(4,))
def _train_step(
    model: Net,
    optimizer: nnx.Optimizer,
    jax_batch: JaxBatch,
    target: Array,
    loss_fn: Callable,
) -> Array:
    """Train for a single step."""

    def f(model: Net, jax_batch: JaxBatch, target: Array, loss_fn: Callable):
        raw_pred: Array = model(jax_batch.states)
        predictions: Array = jnp.take_along_axis(raw_pred, jax_batch.actions, axis=1)
        return loss_fn(predictions, target).mean()

    grad_fn = nnx.value_and_grad(f, has_aux=False)
    loss, grads = grad_fn(model, jax_batch, target, loss_fn)
    optimizer.update(grads)

    return loss


class FlaxNnxPolicyNet(PolicyNet):
    """
    Implements protocal `PolicyNet` with flax.nnx (see rl_2048/dqn/protocols.py)
    """

    def __init__(
        self,
        network_version: str,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        training_params: Optional[TrainingParameters] = None,
    ):
        self.policy_net: Net = _load_predefined_net(
            network_version, in_features, out_features, rngs
        )

        self.training: Optional[TrainingElements]
        if training_params is None:
            self.training = None
        else:
            target_net: Net = _load_predefined_net(
                network_version, in_features, out_features, rngs
            )
            self.training = TrainingElements(
                training_params, self.policy_net, target_net
            )

    def predict(self, state_feature: Sequence[float]) -> PolicyNetOutput:
        state_array: Array = jnp.array(np.array(state_feature))[None, :]
        self.policy_net.eval()
        raw_values: Array = self.policy_net(state_array)[0]
        self.policy_net.train()

        best_action: int = jnp.argmax(raw_values).item()
        best_value: float = raw_values[best_action].item()
        return PolicyNetOutput(best_value, Action(best_action))

    def not_training_error_msg(self) -> str:
        return (
            "TorchPolicyNet is not initailized with training_params. "
            "This function is not supported."
        )

    def optimize(self, batch: Batch) -> Metrics:
        if self.training is None:
            raise ValueError(self.not_training_error_msg())

        jax_batch: JaxBatch = to_jax_batch(batch)
        next_value_predictions: Array = self.training.target_net(jax_batch.next_states)
        next_state_values: Array = next_value_predictions.max(axis=1, keepdims=True)
        expected_state_action_values: Array = jax_batch.rewards + (
            self.training.params.gamma * next_state_values
        ) * (1.0 - jax_batch.games_over)

        loss: Array = _train_step(
            self.policy_net,
            self.training.state,
            jax_batch,
            expected_state_action_values,
            self.training.loss_fn,
        )

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        tau: float = self.training.params.TAU
        target_net_state = tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            nnx.state(self.policy_net),
            nnx.state(self.training.target_net),
        )

        nnx.update(self.training.target_net, target_net_state)

        step: int = self.training.state.step.value
        lr: float = self.training.lr_scheduler(step)

        return {"loss": loss.item(), "step": step, "lr": lr}

    def save(self, model_path: str) -> str:
        if self.training is None:
            raise ValueError(self.not_training_error_msg())

        state: nnx.State = nnx.state(self.policy_net)
        saved_path: str = save_checkpoint(
            ckpt_dir=os.path.abspath(model_path),
            target=state,
            step=self.training.state.step.value,
            keep=10,
        )
        return saved_path

    def load(self, model_path: str):
        state = nnx.state(self.policy_net)
        # Load the parameters
        state = restore_checkpoint(
            ckpt_dir=os.path.dirname(model_path),
            target=state,
        )
        # update the model with the loaded state
        nnx.update(self.policy_net, state)

        if self.training is not None:
            self.training.state.step.value = 0
            nnx.update(self.training.target_net, state)
