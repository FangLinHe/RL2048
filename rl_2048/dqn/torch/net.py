from typing import Union

import torch
import torch.nn as nn
from jaxtyping import Array

from rl_2048.dqn.common import (
    Action,
    Batch,
    Metrics,
    PolicyNetOutput,
    TrainingParameters,
)


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


PREDEFINED_NETWORKS: set[str] = {
    "layers_1024_512_256",
    "layers_512_512_residual_0_128",
    "layers_512_256_128_residual_0_64_32",
    "layers_512_256_256_residual_0_128_128",
}


class TorchPolicyNet:
    policy_net: Net
    target_net: Net
    training_params: TrainingParameters
    loss_fn: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    step_count: int

    def __init__(
        self,
        network_version: str,
        in_features: int,
        out_features: int,
        training_params: TrainingParameters,
    ):
        self.policy_net, self.target_net = self._load_nets(
            network_version, in_features, out_features
        )
        self.training_params = training_params
        self.loss_fn = getattr(nn, training_params.loss_fn)()
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), training_params.lr, amsgrad=True
        )
        self.scheduler = self._load_lr_scheduler(
            training_params.lr,
            training_params.lr_decay_milestones,
            training_params.lr_gamma,
        )
        self.step_count = 0

    def _load_nets(
        self, network_version: str, in_features: int, out_features: int
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

    def _load_lr_scheduler(
        self,
        lr: float,
        lr_decay_milestones: Union[int, list[int]],
        lr_gamma: Union[float, list[float]],
    ) -> torch.optim.lr_scheduler.LRScheduler:
        def gamma_fn(step: int):
            if step in boundaries_and_scales:
                return boundaries_and_scales[step]
            return 1.0

        scheduler = torch.optim.lr_scheduler.LRScheduler
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
                    self.training_params.lr_decay_milestones,
                    self.training_params.lr_gamma,
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

    def predict(self, feature: Array) -> PolicyNetOutput:
        """Predict best action given a feature array.

        Args:
            feature (Array): A feature array (not a batch).

        Returns:
            PolicyNetOutput: Output of policy net (best action and its expected value)
        """
        torch_tensor: torch.Tensor = torch.tensor(feature).view((1, -1))
        training_mode: bool = self.policy_net.training
        self.policy_net.eval()
        best_value, best_action = self.policy_net.forward(torch_tensor).max(1)
        self.policy_net.train(training_mode)
        return PolicyNetOutput(best_value.item(), Action(best_action.item()))

    def optimize(self, batch: Batch) -> Metrics:
        def compute_loss() -> torch.Tensor:
            state_action_values = self.policy_net(batch.states).gather(1, batch.actions)
            with torch.no_grad():
                next_state_values = (
                    self.target_net(batch.next_states).max(1).values.view((-1, 1))
                )

            expected_state_action_values: torch.Tensor = batch.rewards + (
                self.training_params.gamma * next_state_values
            ) * batch.games_over.logical_not().type_as(batch.rewards)
            loss: torch.Tensor = self.loss_fn(
                state_action_values, expected_state_action_values
            )

            return loss

        def optimize_step(loss: torch.Tensor):
            self.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.step_count += 1

        def soft_update():
            """Soft update of the target network's weights

            θ′ ← τ θ + (1 −τ )θ′
            """
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * self.training_params.TAU + target_net_state_dict[key] * (
                    1 - self.training_params.TAU
                )
            self.target_net.load_state_dict(target_net_state_dict)

        step: int = self.step_count
        lr: float = self.scheduler.get_last_lr()[0]

        loss: torch.Tensor = compute_loss()
        optimize_step(loss)
        soft_update()

        return {"loss": loss.item(), "step": step, "lr": lr}
