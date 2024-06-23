import math
from collections.abc import Sequence
from dataclasses import dataclass
from random import SystemRandom
from typing import Optional

from rl_2048.dqn.common import (
    Action,
    Batch,
    DQNParameters,
    Metrics,
    PolicyNetOutput,
)
from rl_2048.dqn.protocols import PolicyNet
from rl_2048.dqn.replay_memory import ReplayMemory, Transition


@dataclass
class TrainingElements:
    params: DQNParameters
    memory: ReplayMemory
    optimize_steps: int = 0


class DQN:
    def __init__(
        self,
        policy_net: PolicyNet,
        dqn_parameters: Optional[DQNParameters] = None,
        output_net_dir: Optional[str] = None,
    ):
        self.policy_net: PolicyNet = policy_net
        self.output_net_dir: Optional[str] = output_net_dir

        self.training: Optional[TrainingElements]
        if dqn_parameters is None:
            self.output_net_dir = None
            self.training = None
        else:
            if output_net_dir is None:
                raise ValueError(
                    "dqn_parameters is not None but output_net_dir is None. "
                    "Please set output_net_dir correctly."
                )
            self.output_net_dir = output_net_dir
            self.training = TrainingElements(
                dqn_parameters, ReplayMemory(dqn_parameters.memory_capacity)
            )

        self._cryptogen: SystemRandom = SystemRandom()

    def predict(self, state: Sequence[float]) -> PolicyNetOutput:
        return self.policy_net.predict(state)

    def _training_none_error_msg(self) -> str:
        return (
            "DQN is not initailized with replay memory parameters. "
            "This function is not supported."
        )

    def get_action_epsilon_greedy(self, state: Sequence[float]) -> Action:
        if self.training is None:
            raise ValueError(self._training_none_error_msg())

        eps_threshold = self.training.params.eps_end + (
            self.training.params.eps_start - self.training.params.eps_end
        ) * math.exp(
            -1.0 * self.training.optimize_steps / self.training.params.eps_decay
        )

        if self._cryptogen.random() > eps_threshold:
            return self.predict(state).action

        return Action(self._cryptogen.randrange(len(Action)))

    def push_transition(self, transition: Transition):
        if self.training is None:
            raise ValueError(self._training_none_error_msg())

        self.training.memory.push(transition)

    def optimize_model(self) -> Optional[Metrics]:
        if self.training is None:
            raise ValueError(self._training_none_error_msg())

        if len(self.training.memory) < self.training.params.batch_size:
            return None

        self.training.optimize_steps += 1

        batch: Batch = self.training.memory.sample(
            min(self.training.params.batch_size, len(self.training.memory))
        )

        return self.policy_net.optimize(batch)

    def save_model(self, filename_prefix: str) -> str:
        return self.policy_net.save(filename_prefix)

    def load_model(self, model_path: str):
        self.policy_net.load(model_path)
