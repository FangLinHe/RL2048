import math
import os
from random import SystemRandom
from typing import List, NamedTuple, Optional, Sequence, Union

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.checkpoints import restore_checkpoint, save_checkpoint
from jax import Array
from jax.tree_util import tree_map

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
    lr: float = 0.001
    lr_decay_milestones: Union[int, List[int]] = 100
    lr_gamma: float = 0.1
    sgd_momentum: float = 0.9

    # for epsilon-greedy algorithm
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 400

    # update rate of the target network
    TAU: float = 0.005

    save_network_steps: int = 1000
    print_loss_steps: int = 100


script_file_path = os.path.dirname(os.path.abspath(__file__))


class PolicyNetOutput(NamedTuple):
    expected_value: float
    action: Action


class DQN:
    def __init__(
        self,
        input_dim: int,
        policy_net: nn.Module,
        output_net_dir: str,
        training_params: TrainingParameters,
        random_key: Array,
    ):
        self.random_key: Array = random_key

        self.policy_net: nn.Module = policy_net
        self.policy_net_train_state: BNTrainState = create_train_state(
            self.random_key,
            self.policy_net,
            input_dim,
            training_params.lr,
            training_params.sgd_momentum,
        )
        self.target_net_train_state: BNTrainState = create_train_state(
            self.random_key,
            self.policy_net,
            input_dim,
            training_params.lr,
            training_params.sgd_momentum,
        )
        self.target_net_train_state = self.target_net_train_state.replace(
            params=self.policy_net_train_state.params,
            batch_stats=self.policy_net_train_state.batch_stats,
        )

        self.output_net_dir: str = output_net_dir

        self.training_params = training_params
        self.memory = ReplayMemory(
            self.random_key, self.training_params.memory_capacity
        )
        self.optimize_steps: int = 0
        self.losses: List[float] = []

        self._cryptogen: SystemRandom = SystemRandom()

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
        eps_threshold = self.training_params.eps_end + (
            self.training_params.eps_start - self.training_params.eps_end
        ) * math.exp(-1.0 * self.optimize_steps / self.training_params.eps_decay)

        if self._cryptogen.random() > eps_threshold:
            return self.get_best_action(state)

        return Action(self._cryptogen.randrange(len(Action)))

    def push_transition(self, transition: Transition):
        self.memory.push(transition)

    def optimize_model(self) -> float:
        if len(self.memory) < self.training_params.batch_size:
            return 0.0

        self.optimize_steps += 1

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
        self.policy_net_train_state, loss = train_step(
            self.policy_net_train_state, batch, expected_state_action_values
        )
        self.losses.append(loss.item())

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

        if self.optimize_steps % self.training_params.print_loss_steps == 0:
            print(
                f"Done optimizing {self.optimize_steps} steps. "
                f"Average loss: {np.mean(self.losses).item()}"
            )
            self.losses = []
        if self.optimize_steps % self.training_params.save_network_steps == 0:
            self.save_model(f"{self.output_net_dir}/step_{self.optimize_steps:04d}")

        return loss.item()

    def save_model(self, root_dir: Optional[str] = None) -> str:
        ckpt_dir: str = (
            os.path.abspath(root_dir) if root_dir is not None else self.output_net_dir
        )
        saved_path: str = save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.policy_net_train_state,
            step=self.optimize_steps,
        )

        return saved_path

    def load_model(self, model_path: str):
        self.policy_net_train_state = restore_checkpoint(
            ckpt_dir=os.path.dirname(model_path), target=self.policy_net_train_state
        )
