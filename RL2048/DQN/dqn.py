import os
import random
from sympy import false
import torch

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
    lr_step_size: int = 100
    lr_gamma: int = 0.1

    # for epsilon-greedy algorithm
    eps_start: float = 0.9
    eps_end: float = 0.1
    eps_decay: float = 0.99

    optimize_times: int = 100


training_params = TrainingParameters()

script_file_path = os.path.dirname(os.path.abspath(__file__))


class DQN:
    def __init__(
        self,
        policy_net: Net,
        training_params: TrainingParameters = TrainingParameters(),
    ):
        self.policy_net: Net = policy_net
        self.training_params = training_params
        self.loss_fn: nn.Module = nn.MSELoss()
        self.optimizer: optim.Optimizer = optim.Adam(
            self.policy_net.parameters(), training_params.lr
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            self.training_params.lr_step_size,
            self.training_params.gamma,
        )
        self.memory = ReplayMemory(self.training_params.memory_capacity)

        self.optimize_steps: int = 0
        self.optimize_count: int = 0

        self.policy_net.train()

    def get_action_epsilon_greedy(self, state: Sequence[float]) -> Action:
        state_tensor: Tensor = torch.tensor(state).view((1, -1))
        eps_threshold = max(
            self.training_params.eps_end,
            self.training_params.eps_start
            * (self.training_params.eps_decay**self.optimize_steps),
        )

        if random.random() > eps_threshold:
            with torch.no_grad():
                _best_value, best_action = self.policy_net(state_tensor).max(1)
            return Action(best_action.item())

        return Action(random.randrange(len(Action)))

    def push_transition(self, transition: Transition) -> bool:
        return self.memory.push(transition)

    def push_transition_and_optimize_automatically(
        self, transition: Transition
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
            print(f"Done. Average loss: {torch.mean(torch.tensor(losses))}")

    def optimize_model(self, reset_memory: bool = True) -> float:
        batch: Batch = self.memory.sample(
            min(self.training_params.batch_size, len(self.memory))
        )

        state_action_values = self.policy_net(batch.states).gather(1, batch.actions)
        all_next_state_action_values = self.policy_net(batch.next_states)
        next_state_max_values = (
            all_next_state_action_values.max(1)[0].detach().view((-1, 1))
        )

        expected_state_action_values = (
            batch.rewards + (
                self.training_params.gamma * next_state_max_values
            ) * batch.games_over.logical_not().type_as(batch.rewards)
        )
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if reset_memory:
            self.memory.reset()
            self.scheduler.step()
            self.optimize_steps += 1

        return loss.item()

    def save_model(self, filename_prefix: str = "policy_net") -> str:
        date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path: str = (
            f"{script_file_path}/../../TrainedNetworks/{filename_prefix}_{date_time_str}.pth"
        )
        torch.save(self.policy_net.state_dict(), save_path)
        print(f"Model saved to path: {save_path}")

        return save_path

    def load_model(self, model_path: str):
        self.policy_net.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
    policy_net = Net(2, 4, [16])
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

    dqn = DQN(policy_net, training_params)

    dqn.push_transition(t1)
    dqn.push_transition(t2)
    dqn.optimize_model()

    print(dqn.get_action_epsilon_greedy(t2.state))

    model_path = dqn.save_model()
    dqn.load_model(model_path)
