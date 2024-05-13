from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F


class Residual(nn.Module):
    def __init__(
        self,
        in_feature_size: int,
        mid_feature_size: int,
        activation_layer: Optional[nn.Module] = nn.ReLU,
    ):
        super(Residual, self).__init__()
        self.linear1 = nn.Linear(in_feature_size, mid_feature_size)
        self.bn1 = nn.BatchNorm1d(num_features=mid_feature_size)
        self.linear2 = nn.Linear(mid_feature_size, mid_feature_size)
        self.bn2 = nn.BatchNorm1d(num_features=mid_feature_size)
        self.linear3 = nn.Linear(mid_feature_size, in_feature_size)
        self.bn3 = nn.BatchNorm1d(num_features=in_feature_size)
        self.activation = activation_layer()

    def forward(self, x):
        y1 = self.activation(self.bn1(self.linear1(x)))
        y2 = self.activation(self.bn2(self.linear2(y1)))
        y3 = self.bn3(self.linear3(y2))
        return x + y3


class Net(nn.Module):
    def __init__(
        self,
        input_feature_size: int,
        output_feature_size: int,
        hidden_layer_sizes: List[int],
        bias: bool = True,
        activation_layer: Optional[nn.Module] = nn.ReLU,
        residual_mid_feature_sizes: List[int] = [],
    ):
        super(Net, self).__init__()
        assert len(residual_mid_feature_sizes) == 0 or len(
            residual_mid_feature_sizes
        ) == len(hidden_layer_sizes)
        in_features = input_feature_size
        layers = []
        for i, out_features in enumerate(hidden_layer_sizes):
            is_residual = False
            if (
                len(residual_mid_feature_sizes) == 0
                or residual_mid_feature_sizes[i] == 0
            ):
                layers.append(nn.Linear(in_features, out_features, bias=bias))
            else:
                assert in_features == out_features
                layers.append(
                    Residual(
                        in_features, residual_mid_feature_sizes[i], activation_layer
                    )
                )
                is_residual = True
            if not is_residual:
                layers.append(nn.BatchNorm1d(num_features=out_features))
            if activation_layer is not None:
                layers.append(activation_layer())
            # if i < len(hidden_layer_sizes) - 1:
            #     layers.append(nn.Dropout())

            in_features = out_features
        layers.append(nn.Linear(in_features, output_feature_size, bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    net = Net(16, 4, [64, 16])
    net.eval()
    input_tensor = torch.rand([1, 16])
    print(f"Output tensor: {net(input_tensor)}")

    net2 = Net(16, 4, [32, 32], residual_mid_feature_sizes=[0, 16])
    net2.eval()
    print(f"Output tensor: {net2(input_tensor)}")
