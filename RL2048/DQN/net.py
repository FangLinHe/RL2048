from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim



class Net(nn.Module):
    def __init__(
            self,
            input_feature_size: int,
            output_feature_size: int,
            hidden_layer_sizes: List[int],
            bias: bool = True,
            activation_layer: Optional[nn.Module] = nn.ReLU
    ):
        super(Net, self).__init__()
        in_features = input_feature_size
        layers = []
        for i, out_features in enumerate(hidden_layer_sizes):
            layers.append(nn.Linear(in_features, out_features, bias=bias))
            layers.append(nn.BatchNorm1d(num_features=out_features))
            if activation_layer is not None:
                layers.append(activation_layer())
            in_features = out_features
        layers.append(nn.Linear(in_features, output_feature_size, bias))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
if __name__ == "__main__":
    net = Net(16, 4, [64, 16])
    input_tensor = torch.rand([1, 16])
    print(f"Output tensor: {net(input_tensor)}")