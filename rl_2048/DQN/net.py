from typing import List, Optional

import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(
        self,
        in_feature_size: int,
        mid_feature_size: int,
        out_feature_size: int,
        activation_layer: nn.Module,
        activation_after_bn: bool = True,
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
        if activation_after_bn:
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
        else:  # activation before bn
            self.block1 = nn.Sequential(
                nn.Linear(in_feature_size, mid_feature_size),
                activation_layer,
                nn.BatchNorm1d(num_features=mid_feature_size),
            )
            self.block2 = nn.Sequential(
                nn.Linear(mid_feature_size, mid_feature_size),
                activation_layer,
                nn.BatchNorm1d(num_features=mid_feature_size),
            )
            self.blocks3 = nn.Sequential(
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
        hidden_layer_sizes: List[int],
        activation_layer: nn.Module,
        bias: bool = True,
        residual_mid_feature_sizes: Optional[List[int]] = None,
    ):
        super(Net, self).__init__()
        if residual_mid_feature_sizes is None:
            residual_mid_feature_sizes = []
        if len(residual_mid_feature_sizes) not in {0, len(hidden_layer_sizes)}:
            raise ValueError(
                "`residual_mid_feature_sizes` should be either None or have the same "
                f"length as `hidden_layer_sizes` ({len(hidden_layer_sizes)}), but got "
                f"({len(residual_mid_feature_sizes)})"
            )
        in_features = input_feature_size
        layers: List[nn.Module] = []

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
                layers.append(nn.Linear(in_features, out_features, bias=bias))
                layers.append(
                    nn.BatchNorm1d(num_features=out_features),
                )

            layers.append(activation_layer)

            in_features = out_features
        layers.append(nn.Linear(in_features, output_feature_size, bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    net = Net(16, 4, [64, 16], nn.ReLU())
    net.eval()
    input_tensor = torch.rand([1, 16])
    print(f"Output tensor: {net(input_tensor)}")

    net2 = Net(16, 4, [32, 32], nn.ReLU(), residual_mid_feature_sizes=[0, 16])
    net2.eval()
    print(f"Output tensor: {net2(input_tensor)}")

    net3 = Net(16, 4, [32, 16], nn.ReLU(), residual_mid_feature_sizes=[0, 8])
    net3.eval()
    print(f"Output tensor: {net3(input_tensor)}")
