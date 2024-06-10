import math
import os
from collections.abc import Sequence
from random import SystemRandom
from typing import Any, Callable, NamedTuple, Optional, Union

import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.checkpoints import PyTree, restore_checkpoint, save_checkpoint
from jax import Array
from jax.tree_util import tree_map
from tensorboardX import SummaryWriter

from rl_2048.jax_dqn.net import (
    BNTrainState,
    create_train_state,
    eval_forward,
    train_step,
)
from rl_2048.jax_dqn.replay_memory import Action, Batch, ReplayMemory, Transition


class TrainingParameters(NamedTuple):
    memory_capacity: int = 1024
    gamma: float = 0.99
    batch_size: int = 64
    optimizer: str = "adamw"
    lr: float = 0.001
    lr_decay_milestones: Union[int, list[int]] = 100
    lr_gamma: Union[float, list[float]] = 0.1
    loss_fn: str = "huber_loss"

    # for epsilon-greedy algorithm
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 400

    # update rate of the target network
    TAU: float = 0.005

    save_network_steps: int = 1000
    print_loss_steps: int = 100
    tb_write_steps: int = 50

    pretrained_net_path: str = ""


script_file_path = os.path.dirname(os.path.abspath(__file__))


class PolicyNetOutput(NamedTuple):
    expected_value: float
    action: Action


def create_learning_rate_fn(training_params: TrainingParameters) -> optax.Schedule:
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


class DQN:
    def __init__(
        self,
        input_dim: int,
        policy_net: nn.Module,
        output_net_dir: str,
        training_params: TrainingParameters,
        random_key: Array,
    ):
        def _make_hparams_dict(params: TrainingParameters):
            hparams: dict[str, Any] = {}
            for k, v in params._asdict().items():
                key = f"hparams/{k}"
                value = (
                    v
                    if not isinstance(v, list)
                    else f"[{', '.join(str(elm) for elm in v)}]"
                )
                hparams[key] = value
            return hparams

        self.random_key: Array = random_key

        self.lr_scheduler: optax.Schedule = create_learning_rate_fn(training_params)
        self.policy_net: nn.Module = policy_net
        self.policy_net_train_state: BNTrainState = create_train_state(
            self.random_key,
            self.policy_net,
            input_dim,
            training_params.optimizer,
            self.lr_scheduler,
        )
        self.target_net_train_state: BNTrainState = create_train_state(
            self.random_key,
            self.policy_net,
            input_dim,
            training_params.optimizer,
            self.lr_scheduler,
        )
        self.target_net_train_state = self.target_net_train_state.replace(
            params=self.policy_net_train_state.params,
            batch_stats=self.policy_net_train_state.batch_stats,
        )

        self.output_net_dir: str = output_net_dir

        self.training_params = training_params
        self.optax_loss_fn = getattr(optax, training_params.loss_fn)
        self.memory = ReplayMemory(
            self.random_key, self.training_params.memory_capacity
        )
        self.optimize_steps: int = 0
        self.losses: list[float] = []

        self._cryptogen: SystemRandom = SystemRandom()

        self.eps_threshold: float = 0.0
        self.summary_writer = SummaryWriter()

        self.summary_writer.add_hparams(_make_hparams_dict(training_params), dict())
        self.summary_writer.add_text("output_net_dir", output_net_dir)

    @staticmethod
    def infer_action_net(
        net_apply: Callable,
        variables: PyTree,
        state: Sequence[float],
    ) -> PolicyNetOutput:
        raw_values: Array = net_apply(
            variables,
            jnp.array(np.array(state))[None, :],
        )[0]
        best_action: int = jnp.argmax(raw_values).item()
        best_value: float = raw_values[best_action].item()
        return PolicyNetOutput(best_value, Action(best_action))

    @staticmethod
    def infer_action(
        policy_net_state: BNTrainState,
        state: Sequence[float],
    ) -> PolicyNetOutput:
        raw_values: Array = eval_forward(
            policy_net_state, jnp.array(np.array(state))[None, :]
        )[0]
        best_action: int = jnp.argmax(raw_values).item()
        best_value: float = raw_values[best_action].item()
        return PolicyNetOutput(best_value, Action(best_action))

    def get_best_action(self, state: Sequence[float]) -> Action:
        best_action = self.infer_action(self.policy_net_train_state, state).action
        return best_action

    def get_action_epsilon_greedy(self, state: Sequence[float]) -> Action:
        self.eps_threshold = self.training_params.eps_end + (
            self.training_params.eps_start - self.training_params.eps_end
        ) * math.exp(-1.0 * self.optimize_steps / self.training_params.eps_decay)

        if self._cryptogen.random() > self.eps_threshold:
            return self.get_best_action(state)

        return Action(self._cryptogen.randrange(len(Action)))

    def push_transition(self, transition: Transition):
        self.memory.push(transition)

    def optimize_model(self, game_iter: int) -> float:
        if len(self.memory) < self.training_params.batch_size:
            return 0.0

        batch: Batch = self.memory.sample(
            min(self.training_params.batch_size, len(self.memory))
        )
        next_value_predictions = eval_forward(
            self.target_net_train_state, batch.next_states
        )
        next_state_values = next_value_predictions.max(axis=1, keepdims=True)
        expected_state_action_values: Array = batch.rewards + (
            self.training_params.gamma * next_state_values
        ) * (1.0 - batch.games_over)
        self.policy_net_train_state, loss, lr = train_step(
            self.policy_net_train_state,
            batch,
            expected_state_action_values,
            self.lr_scheduler,
            self.optax_loss_fn,
        )
        loss_val: float = loss.item()
        self.losses.append(loss_val)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        tau: float = self.training_params.TAU
        target_net_params = tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            self.policy_net_train_state.params,
            self.target_net_train_state.params,
        )
        target_net_batch_stats = tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            self.policy_net_train_state.batch_stats,
            self.target_net_train_state.batch_stats,
        )
        self.target_net_train_state = self.target_net_train_state.replace(
            params=target_net_params, batch_stats=target_net_batch_stats
        )

        self.optimize_steps += 1

        if self.optimize_steps % self.training_params.tb_write_steps == 0:
            self.summary_writer.add_scalar(
                "train/game_iter", game_iter, self.optimize_steps
            )
            self.summary_writer.add_scalar(
                "train/eps_thresh", self.eps_threshold, self.optimize_steps
            )
            self.summary_writer.add_scalar("train/lr", lr, self.optimize_steps)
            self.summary_writer.add_scalar("train/loss", loss_val, self.optimize_steps)
            self.summary_writer.add_scalar(
                "train/memory_size", len(self.memory), self.optimize_steps
            )
        if self.optimize_steps % self.training_params.print_loss_steps == 0:
            print(
                f"Done optimizing {self.optimize_steps} steps. "
                f"Average loss: {np.mean(self.losses).item()}"
            )
            self.losses = []
        if self.optimize_steps % self.training_params.save_network_steps == 0:
            self.save_model(f"{self.output_net_dir}")

        return loss.item()

    def save_model(self, root_dir: Optional[str] = None) -> str:
        ckpt_dir: str = (
            os.path.abspath(root_dir) if root_dir is not None else self.output_net_dir
        )
        saved_path: str = save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.policy_net_train_state,
            step=self.optimize_steps,
            keep=10,
        )

        return saved_path

    def load_model(self, model_path: str):
        self.policy_net_train_state = restore_checkpoint(
            ckpt_dir=os.path.dirname(model_path), target=self.policy_net_train_state
        )
        # Reset step to 0, so LR scheduler works as expected
        self.policy_net_train_state = self.policy_net_train_state.replace(step=0)
        self.target_net_train_state = self.target_net_train_state.replace(
            params=self.policy_net_train_state.params,
            batch_stats=self.policy_net_train_state.batch_stats,
        )
