from email.policy import Policy
import os
import random
import torch
import math

from datetime import datetime
from typing import NamedTuple, List, Sequence, Union
from .net import Net
from .replay_memory import Action, Batch, ReplayMemory, Transition
from torch import Tensor, nn, optim


class TrainingParameters(NamedTuple):
    memory_capacity: int = 1024
    gamma: float = 0.99
    batch_size: int = 64
    lr: float = 0.001
    lr_decay_milestones: Union[int, List[int]] = 100
    lr_gamma: int = 0.1

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
        policy_net: Net,
        target_net: Net,
        output_net_dir: str,
        training_params: TrainingParameters = TrainingParameters(),
    ):
        self.policy_net: Net = policy_net
        self.target_net: Net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.output_net_dir: str = output_net_dir

        self.training_params = training_params
        self.loss_fn: nn.Module = nn.HuberLoss()  # nn.MSELoss()
        self.optimizer: optim.Optimizer = optim.AdamW(
            self.policy_net.parameters(), training_params.lr, amsgrad=True
        )
        if isinstance(self.training_params.lr_decay_milestones, int):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                self.training_params.lr_decay_milestones,
                self.training_params.gamma,
            )
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                self.training_params.lr_decay_milestones,
                self.training_params.gamma,
            )
        self.memory = ReplayMemory(self.training_params.memory_capacity)

        self.optimize_steps: int = 0

        self.policy_net.train()

        self.losses: List[float] = []

    @staticmethod
    def infer_action(policy_net: nn.Module, state: Sequence[float]) -> PolicyNetOutput:
        state_tensor: Tensor = torch.tensor(state).view((1, -1))
        best_value, best_action = policy_net(state_tensor).max(1)
        return PolicyNetOutput(best_value.item(), Action(best_action.item()))

    def get_best_action(self, state: Sequence[float]) -> Action:
        self.policy_net.eval()
        best_action = self.infer_action(self.policy_net, state).action
        self.policy_net.train()
        return best_action


    def get_action_epsilon_greedy(self, state: Sequence[float]) -> Action:
        eps_threshold = self.training_params.eps_end + (
            self.training_params.eps_start - self.training_params.eps_end
        ) * math.exp(-1.0 * self.optimize_steps / self.training_params.eps_decay)

        if random.random() > eps_threshold:
            return self.get_best_action(state)

        return Action(random.randrange(len(Action)))

    def push_transition(self, transition: Transition):
        self.memory.push(transition)

    def optimize_model(self) -> float:
        if len(self.memory) < self.training_params.batch_size:
            return 0.0

        self.optimize_steps += 1

        batch: Batch = self.memory.sample(
            min(self.training_params.batch_size, len(self.memory))
        )

        state_action_values = self.policy_net(batch.states).gather(1, batch.actions)
        with torch.no_grad():
            next_state_values = (
                self.target_net(batch.next_states).max(1).values.view((-1, 1))
            )

        expected_state_action_values = batch.rewards + (
            self.training_params.gamma * next_state_values
        ) * batch.games_over.logical_not().type_as(batch.rewards)
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.training_params.TAU + target_net_state_dict[key] * (
                1 - self.training_params.TAU
            )
        self.target_net.load_state_dict(target_net_state_dict)

        if self.optimize_steps % self.training_params.print_loss_steps == 0:
            print(
                f"Done optimizing {self.optimize_steps} steps. "
                f"Average loss: {torch.tensor(self.losses).mean().item()}"
            )
            self.losses = []
        if self.optimize_steps % self.training_params.save_network_steps == 0:
            self.save_model(f"{self.output_net_dir}/step_{self.optimize_steps:04d}")

        return loss.item()

    def save_model(self, filename_prefix: str = "policy_net") -> str:
        save_path: str = f"{filename_prefix}.pth"
        torch.save(self.policy_net.state_dict(), save_path)
        # print(f"Model saved to path: {save_path}")

        return save_path

    def load_model(self, model_path: str):
        self.policy_net.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
    policy_net = Net(2, 4, [16])
    target_net = Net(2, 4, [16])
    training_params = TrainingParameters(
        memory_capacity=1024,
        gamma=0.99,
        batch_size=64,
        lr=0.001,
        eps_start=0.0,
        eps_end=0.0,
    )
    t1 = Transition(
        state=[1.0, 0.5],
        action=Action.UP,
        next_state=[2.0, 0.0],
        reward=10.0,
        game_over=False,
    )
    t2 = Transition(
        state=[2.0, 0.0],
        action=Action.LEFT,
        next_state=[-0.5, 1.0],
        reward=-1.0,
        game_over=False,
    )

    dqn = DQN(policy_net, target_net, training_params)

    dqn.push_transition(t1)
    dqn.push_transition(t2)
    dqn.optimize_model()

    print(dqn.get_action_epsilon_greedy(t2.state))

    model_path = dqn.save_model()
    dqn.load_model(model_path)
