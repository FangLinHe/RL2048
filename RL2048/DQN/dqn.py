import os
import random
import torch
import math

from datetime import datetime
from typing import NamedTuple, List, Sequence
from .net import Net
from .replay_memory import Action, Batch, ReplayMemory, Transition
from torch import Tensor, nn, optim


class TrainingParameters(NamedTuple):
    memory_capacity: int = 1024
    gamma: float = 0.99
    batch_size: int = 64
    lr: float = 0.001
    lr_step_sizes: List[int] = [100, 80, 60]
    lr_gamma: int = 0.1

    # for epsilon-greedy algorithm
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 400

    optimize_times: int = 100

    # update rate of the target network
    TAU: float = 0.005


script_file_path = os.path.dirname(os.path.abspath(__file__))


class DQN:
    def __init__(
        self,
        policy_net: Net,
        target_net: Net,
        training_params: TrainingParameters = TrainingParameters(),
    ):
        self.policy_net: Net = policy_net
        self.target_net: Net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.training_params = training_params
        self.loss_fn: nn.Module = nn.SmoothL1Loss()
        self.optimizer: optim.Optimizer = optim.Adam(
            self.policy_net.parameters(), training_params.lr
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            self.training_params.lr_step_sizes,
            self.training_params.gamma,
        )
        self.memory = ReplayMemory(self.training_params.memory_capacity)

        self.optimize_steps: int = 0
        self.optimize_count: int = 0

        self.policy_net.train()

    def get_action_epsilon_greedy(self, state: Sequence[float]) -> Action:
        state_tensor: Tensor = torch.tensor(state).view((1, -1))
        eps_threshold = self.training_params.eps_end + math.exp(
            -1.0 * self.optimize_steps / self.training_params.eps_decay
        )

        if random.random() > eps_threshold:
            with torch.no_grad():
                _best_value, best_action = self.policy_net(state_tensor).max(1)
            return Action(best_action.item())

        return Action(random.randrange(len(Action)))

    def push_transition(self, transition: Transition) -> bool:
        return self.memory.push(transition)

    def push_transition_and_optimize_automatically(
        self, transition: Transition, output_net_dir: str
    ) -> bool:
        self.push_transition(transition)
        if self.memory.is_full():
            losses: List[float] = []
            self.optimize_count += 1
            print(f"Optimizing - {self.optimize_count}...")
            for i in range(self.training_params.optimize_times):
                is_last_round = i == self.training_params.optimize_times - 1
                loss = self.optimize_model(reset_memory=is_last_round)
                losses.append(loss)
            print(f"Done. Average loss: {torch.tensor(losses).mean().item()}")
            self.save_model(f"{output_net_dir}/step_{self.optimize_count:04d}")

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

    def optimize_model(self, reset_memory: bool = True) -> float:
        batch: Batch = self.memory.sample(
            min(self.training_params.batch_size, len(self.memory))
        )

        state_action_values = self.policy_net(batch.states).gather(1, batch.actions)
        with torch.no_grad():
            next_state_values = self.target_net(batch.next_states).max(1).values.view((-1, 1))

        expected_state_action_values = batch.rewards + (
            self.training_params.gamma * next_state_values
        ) * batch.games_over.logical_not().type_as(batch.rewards)
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        if reset_memory:
            self.memory.reset()
            self.scheduler.step()
            self.optimize_steps += 1

        return loss.item()

    def save_model(self, filename_prefix: str = "policy_net") -> str:
        save_path: str = f"{filename_prefix}.pth"
        torch.save(self.policy_net.state_dict(), save_path)
        print(f"Model saved to path: {save_path}")

        return save_path

    def load_model(self, model_path: str):
        self.policy_net.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
    policy_net = Net(2, 4, [16])
    target_net = Net(2, 4, [16])
    training_params = TrainingParameters(
        memory_capacity=1024, gamma=0.99, batch_size=64, lr=0.001
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
