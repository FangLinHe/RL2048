from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from rl_2048.dqn.common import Batch, TrainingParameters


class JaxBatch(NamedTuple):
    states: Array
    actions: Array
    next_states: Array
    rewards: Array
    games_over: Array


def to_jax_batch(batch: Batch) -> JaxBatch:
    return JaxBatch(
        states=jnp.array(np.array(batch.states)),
        actions=jnp.array(np.array(batch.actions), dtype=jnp.int32).reshape((-1, 1)),
        next_states=jnp.array(np.array(batch.next_states)),
        rewards=jnp.array(np.array(batch.rewards)).reshape((-1, 1)),
        games_over=jnp.array(np.array(batch.games_over)).reshape((-1, 1)),
    )


def _create_lr_scheduler(training_params: TrainingParameters) -> optax.Schedule:
    """Creates learning rate schedule."""
    lr_scheduler_fn: optax.Schedule
    if isinstance(training_params.lr_decay_milestones, int):
        if not isinstance(training_params.lr_gamma, float):
            raise ValueError(
                "Type of `lr_gamma` should be float, but got "
                f"{type(training_params.lr_gamma)}."
            )
        lr_scheduler_fn = optax.exponential_decay(
            init_value=training_params.lr,
            transition_steps=training_params.lr_decay_milestones,
            decay_rate=training_params.lr_gamma,
            staircase=True,
        )
    elif len(training_params.lr_decay_milestones) > 0:
        boundaries_and_scales: dict[int, float]
        if isinstance(training_params.lr_gamma, float):
            boundaries_and_scales = {
                step: training_params.lr_gamma
                for step in training_params.lr_decay_milestones
            }
        else:
            gamma_len = len(training_params.lr_gamma)
            decay_len = len(training_params.lr_decay_milestones)
            if gamma_len != decay_len:
                raise ValueError(
                    f"Lengths of `lr_gamma` ({gamma_len}) should be the same as "
                    f"`lr_decay_milestones` ({decay_len})"
                )
            boundaries_and_scales = {
                step: gamma
                for step, gamma in zip(
                    training_params.lr_decay_milestones, training_params.lr_gamma
                )
            }

        lr_scheduler_fn = optax.piecewise_constant_schedule(
            init_value=training_params.lr, boundaries_and_scales=boundaries_and_scales
        )
    else:
        lr_scheduler_fn = optax.constant_schedule(training_params.lr)

    return lr_scheduler_fn
