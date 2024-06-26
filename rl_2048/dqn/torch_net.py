from collections.abc import Iterable, Sequence
from typing import Optional, Union

import torch
import torch.nn as nn

from rl_2048.dqn.common import (
    PREDEFINED_NETWORKS,
    Action,
    Batch,
    Metrics,
    PolicyNetOutput,
    TrainingParameters,
)
from rl_2048.dqn.protocols import PolicyNet


class Residual(nn.Module):
    def __init__(
        self,
        in_feature_size: int,
        mid_feature_size: int,
        out_feature_size: int,
        activation_layer: nn.Module,
    ):
        # Two possibilities:
        # 1. Input / output feature sizes are the same, e.g.:
        #    x - (Linear 512x128) - (Linear 128x128) - (Linear 128x512) - sum - y
        #     \------------------------------------------------------------/
        # 2. Input / output feature sizes are different, e.g.:
        #    x - (Linear 512x64) - (Linear 64x64) - (Linear 64x256) - sum - y
        #     \--------------------------------------------(AvgPool(2))----/
        super(Residual, self).__init__()
        if in_feature_size % out_feature_size != 0:
            raise ValueError(
                f"in_feature_size ({in_feature_size}) must be divisible by "
                f"out_feature_size ({out_feature_size})"
            )
        self.block1: nn.Module
        self.block2: nn.Module
        self.block3: nn.Module
        self.block1 = nn.Sequential(
            nn.Linear(in_feature_size, mid_feature_size),
            nn.BatchNorm1d(num_features=mid_feature_size),
            activation_layer,
        )
        self.block2 = nn.Sequential(
            nn.Linear(mid_feature_size, mid_feature_size),
            nn.BatchNorm1d(num_features=mid_feature_size),
            activation_layer,
        )
        self.block3 = nn.Sequential(
            nn.Linear(mid_feature_size, out_feature_size),
            nn.BatchNorm1d(num_features=out_feature_size),
        )
        self.pool_or_identity = (
            nn.AvgPool1d(in_feature_size // out_feature_size)
            if in_feature_size != out_feature_size
            else nn.Identity()
        )

    def forward(self, x):
        y = self.block3(self.block2(self.block1(x)))
        return self.pool_or_identity(x) + y


class Net(nn.Module):
    def __init__(
        self,
        input_feature_size: int,
        output_feature_size: int,
        hidden_layer_sizes: list[int],
        activation_layer: nn.Module,
        residual_mid_feature_sizes: list[int],
    ):
        super(Net, self).__init__()
        if len(residual_mid_feature_sizes) not in {0, len(hidden_layer_sizes)}:
            raise ValueError(
                "`residual_mid_feature_sizes` should be either None or have the same "
                f"length as `hidden_layer_sizes` ({len(hidden_layer_sizes)}), but got "
                f"({len(residual_mid_feature_sizes)})"
            )
        in_features = input_feature_size
        layers: list[nn.Module] = []

        is_residual = len(residual_mid_feature_sizes) > 0
        for i, out_features in enumerate(hidden_layer_sizes):
            if is_residual and residual_mid_feature_sizes[i] != 0:
                layers.append(
                    Residual(
                        in_features,
                        residual_mid_feature_sizes[i],
                        out_features,
                        activation_layer,
                    )
                )
            else:
                layers.append(nn.Linear(in_features, out_features, bias=True))
                layers.append(
                    nn.BatchNorm1d(num_features=out_features),
                )

            layers.append(activation_layer)

            in_features = out_features
        layers.append(nn.Linear(in_features, output_feature_size, True))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TrainingElements:
    """Class for keeping track of training variables"""

    def __init__(
        self,
        net_params: Iterable[nn.parameter.Parameter],
        training_params: TrainingParameters,
    ):
        self.params: TrainingParameters = training_params
        self.loss_fn: nn.Module = getattr(nn, self.params.loss_fn)()
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            net_params, self.params.lr, amsgrad=True
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = self._load_lr_scheduler(
            self.params.lr_decay_milestones,
            self.params.lr_gamma,
        )
        self.step_count: int = 0

    def _load_lr_scheduler(
        self,
        lr_decay_milestones: Union[int, list[int]],
        lr_gamma: Union[float, list[float]],
    ) -> torch.optim.lr_scheduler.LRScheduler:
        def gamma_fn(step: int):
            if step in boundaries_and_scales:
                return boundaries_and_scales[step]
            return 1.0

        scheduler: torch.optim.lr_scheduler.LRScheduler
        # decay LR by gamma after every N steps
        if isinstance(lr_decay_milestones, int):
            if not isinstance(lr_gamma, float):
                raise ValueError(
                    "Type of `lr_gamma` should be float, but got " f"{type(lr_gamma)}."
                )
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                lr_decay_milestones,
                lr_gamma,
            )
        elif len(lr_decay_milestones) > 0:
            # decay LR by gamma after each milestone
            if isinstance(lr_gamma, float):
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    lr_decay_milestones,
                    lr_gamma,
                )
            else:  # lr_gamma is a list
                gamma_len = len(lr_gamma)
                decay_len = len(lr_decay_milestones)
                if gamma_len != decay_len:
                    raise ValueError(
                        f"Lengths of `lr_gamma` ({gamma_len}) should be the same as "
                        f"`lr_decay_milestones` ({decay_len})"
                    )
                boundaries_and_scales = {
                    step: gamma for step, gamma in zip(lr_decay_milestones, lr_gamma)
                }
                scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                    self.optimizer,
                    gamma_fn,
                )
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

        return scheduler


def load_nets(
    network_version: str, in_features: int, out_features: int
) -> tuple[Net, Net]:
    hidden_layers: list[int]
    residual_mid_feature_sizes: list[int]
    if network_version == "layers_1024_512_256":
        hidden_layers = [1024, 512, 256]
        residual_mid_feature_sizes = []
    elif network_version == "layers_512_512_residual_0_128":
        hidden_layers = [512, 512]
        residual_mid_feature_sizes = [0, 128]
    elif network_version == "layers_512_256_128_residual_0_64_32":
        hidden_layers = [512, 256, 128]
        residual_mid_feature_sizes = [0, 64, 32]
    elif network_version == "layers_512_256_256_residual_0_128_128":
        hidden_layers = [512, 256, 256]
        residual_mid_feature_sizes = [0, 128, 128]
    else:
        raise NameError(
            f"Network version {network_version} not in {PREDEFINED_NETWORKS}."
        )

    policy_net = Net(
        in_features,
        out_features,
        hidden_layers,
        nn.ReLU(),
        residual_mid_feature_sizes=residual_mid_feature_sizes,
    )
    target_net = Net(
        in_features,
        out_features,
        hidden_layers,
        nn.ReLU(),
        residual_mid_feature_sizes=residual_mid_feature_sizes,
    )
    return (policy_net, target_net)


class TorchPolicyNet(PolicyNet):
    """
    Implements protocal `PolicyNet` with PyTorch (see rl_2048/dqn/protocols.py)
    """

    policy_net: Net
    target_net: Net
    training: Optional[TrainingElements]

    def __init__(
        self,
        network_version: str,
        in_features: int,
        out_features: int,
        training_params: Optional[TrainingParameters] = None,
    ):
        self.policy_net, self.target_net = load_nets(
            network_version, in_features, out_features
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

        if training_params is None:
            self.training = None
        else:
            self.training = TrainingElements(
                self.policy_net.parameters(), training_params
            )

    def predict(self, state_feature: Sequence[float]) -> PolicyNetOutput:
        """Predict best action given a feature array.

        Args:
            feature (Array): A feature array (not a batch).

        Returns:
            PolicyNetOutput: Output of policy net (best action and its expected value)
        """
        state_tensor: torch.Tensor = torch.tensor(state_feature).view((1, -1))
        training_mode: bool = self.policy_net.training
        self.policy_net.eval()
        best_value, best_action = self.policy_net.forward(state_tensor).max(1)
        self.policy_net.train(training_mode)
        return PolicyNetOutput(best_value.item(), Action(best_action.item()))

    def optimize(self, batch: Batch) -> Metrics:
        def error_msg() -> str:
            return (
                "TorchPolicyNet is not initailized with training_params. "
                "This function is not supported."
            )

        def compute_loss(training: TrainingElements) -> torch.Tensor:
            states: torch.Tensor = torch.tensor(batch.states)
            actions: torch.Tensor = torch.tensor(batch.actions, dtype=torch.int64).view(
                (-1, 1)
            )
            next_states: torch.Tensor = torch.tensor(batch.next_states)
            rewards: torch.Tensor = torch.tensor(batch.rewards).view((-1, 1))
            games_over: torch.Tensor = torch.tensor(
                batch.games_over, dtype=torch.bool
            ).view((-1, 1))

            state_action_values = self.policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_state_values = (
                    self.target_net(next_states).max(1).values.view((-1, 1))
                )

            gamma = training.params.gamma
            expected_state_action_values: torch.Tensor = rewards + (
                gamma * next_state_values
            ) * games_over.logical_not().type_as(rewards)
            loss: torch.Tensor = training.loss_fn(
                state_action_values, expected_state_action_values
            )

            return loss

        def optimize_step(training: TrainingElements, loss: torch.Tensor):
            training.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
            training.optimizer.step()
            training.scheduler.step()
            training.step_count += 1

        def soft_update(training: TrainingElements):
            """Soft update of the target network's weights

            θ′ ← τ θ + (1 −τ )θ′
            """

            tau = training.params.TAU
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * tau + target_net_state_dict[key] * (1 - tau)
            self.target_net.load_state_dict(target_net_state_dict)

        if self.training is None:
            raise ValueError(error_msg())

        lr: float = self.training.scheduler.get_last_lr()[0]
        loss: torch.Tensor = compute_loss(self.training)
        optimize_step(self.training, loss)
        soft_update(self.training)
        step: int = self.training.step_count

        return {"loss": loss.item(), "step": step, "lr": lr}

    def save(self, model_path: str) -> str:
        if not model_path.endswith(".pth"):
            model_path = f"{model_path}.pth"
        torch.save(self.policy_net.state_dict(), model_path)
        return model_path

    def load(self, model_path: str):
        self.policy_net.load_state_dict(torch.load(model_path))
