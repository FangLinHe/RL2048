import os
import random
import torch

from datetime import datetime
from typing import NamedTuple, Sequence
from .net import Net
from .replay_memory import Action, Batch, ReplayMemory, Transition
from torch import Tensor, nn, optim

class TrainingParameters(NamedTuple):
    memory_capacity: int = 1024
    gamma: float = 0.99
    batch_size: int = 64
    lr: float = 0.001

    # for epsilon-greedy algorithm
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay: float = 0.95

training_params = TrainingParameters()

script_file_path = os.path.dirname(os.path.abspath(__file__))

class DQN:
    def __init__(self, policy_net: Net, training_params: TrainingParameters = TrainingParameters()):
        self.policy_net: Net = policy_net
        self.loss_fn: nn.Module = nn.MSELoss()
        self.optimizer: optim.Optimizer = optim.Adam(
            self.policy_net.parameters(),
            training_params.lr
        )
        self.training_params = training_params
        self.memory = ReplayMemory(self.training_params.memory_capacity)

        self.optimize_steps: int = 0

        self.policy_net.train()

    def next_action_epsilon_greedy(self, state: Sequence[float]) -> Action:
        state_tensor: Tensor = torch.tensor(state).view((1, -1))
        eps_threshold = max(
            self.training_params.eps_end,
            self.training_params.eps_start * (self.training_params.eps_decay ** self.optimize_steps)
        )
        
        if random.random() > eps_threshold:
            with torch.no_grad():
                _best_value, best_action = policy_net(state_tensor).max(1)
            return Action(best_action[0, 0])
        
        return Action(random.randrange(len(Action)))

    def push_transition(self, transition: Transition) -> bool:
        return self.memory.push(transition)
    
    def optimize_model(self):
        batch: Batch = self.memory.sample(min(self.training_params.batch_size, len(self.memory)))

        state_action_values = self.policy_net(batch.states).gather(1, batch.actions)
        all_next_state_action_values = self.policy_net(batch.next_states)
        next_state_max_values = all_next_state_action_values.max(1)[0].detach().view((-1, 1))

        expected_state_action_values = batch.rewards + self.training_params.gamma * next_state_max_values
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.reset()
        self.optimize_steps += 1

    def save_model(self, filename_prefix: str = "policy_net") -> str:
        date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path: str = f"{script_file_path}/../../TrainedNetworks/{filename_prefix}_{date_time_str}.pth"
        torch.save(self.policy_net.state_dict(), save_path)
        print(f"Model saved to path: {save_path}")

        return save_path
    
    def load_model(self, model_path: str):
        self.policy_net.load_state_dict(torch.load(model_path))

if __name__ == "__main__":
    policy_net = Net(2, 4, [16])
    training_params = TrainingParameters(
        memory_capacity = 1024,
        gamma = 0.99,
        batch_size = 64,
        lr = 0.001
    )
    t1 = Transition(
        state=[1.0, 0.5],
        action=Action.UP,
        next_state=[2.0, 0.0],
        reward=10.0
    )
    t2 = Transition(
        state=[2.0, 0.0],
        action=Action.LEFT,
        next_state=[-0.5, 1.0],
        reward=-1.0
    )

    dqn = DQN(policy_net, training_params)
    
    dqn.push_transition(t1)
    dqn.push_transition(t2)
    dqn.optimize_model()

    print(dqn.next_action_epsilon_greedy(t2.state))

    model_path = dqn.save_model()
    dqn.load_model(model_path)