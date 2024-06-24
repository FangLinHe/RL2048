"""
Implement the following protocol

class PolicyNet(Protocol):
    def predict(self, feature: Sequence[float]) -> PolicyNetOutput: ...

    def optimize(self, batch: Batch) -> Metrics: ...

    def save(self, filename_prefix: str) -> str: ...

    def load(self, model_path: str): ...
"""

from typing import Callable, Union

from flax import nnx
from jaxtyping import Array

from rl_2048.dqn.common import PREDEFINED_NETWORKS


class ResidualBlock(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
        activation_fn: Callable,
        rngs: nnx.Rngs,
    ):
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.activation_fn = activation_fn

        self.linear1 = nnx.Linear(in_dim, mid_dim, use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(mid_dim, rngs=rngs)
        self.linear2 = nnx.Linear(mid_dim, mid_dim, use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(mid_dim, rngs=rngs)
        self.linear3 = nnx.Linear(mid_dim, out_dim, use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(out_dim, rngs=rngs)

    def __call__(self, x: Array):
        residual: Array = x
        x = self.bn1(self.linear1(x))
        x = self.activation_fn(x)
        x = self.activation_fn(self.bn2(self.linear2(x)))
        x = self.bn3(self.linear3(x))

        if residual.shape != x.shape:
            pool_size: int = self.in_dim // self.out_dim
            print(residual.shape)
            residual = nnx.avg_pool(
                residual[:, :, None],
                window_shape=(
                    1,
                    pool_size,
                ),
                strides=(
                    1,
                    pool_size,
                ),
            )[:, :, 0]

        return x + residual


class Net(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        net_activation_fn: Callable,
        residual_mid_dims: tuple[int, ...],
        rngs: nnx.Rngs,
    ):
        if len(residual_mid_dims) == 0:
            residual_mid_dims = tuple(0 for _ in range(len(hidden_dims)))

        def validate_args():
            N_hidden, N_res = len(hidden_dims), len(residual_mid_dims)
            if N_hidden != N_res:
                raise ValueError(
                    "`residual_mid_dims` should be either empty or have the same "
                    f"length as `hidden_dims` ({N_hidden}), but got ({N_res})"
                )

        validate_args()

        layers: list[nnx.Module] = []
        for residual_mid_dim, hidden_dim in zip(residual_mid_dims, hidden_dims):
            block: list[Union[nnx.Module, Callable]] = []
            if residual_mid_dim == 0:
                block.append(nnx.Linear(in_dim, hidden_dim, use_bias=False, rngs=rngs))
                block.append(nnx.BatchNorm(hidden_dim, rngs=rngs))
            else:
                block.append(
                    ResidualBlock(
                        in_dim, residual_mid_dim, hidden_dim, net_activation_fn, rngs
                    )
                )
            in_dim = hidden_dim
            block.append(net_activation_fn)
            layers.append(nnx.Sequential(*block))

        layers.append(nnx.Linear(in_dim, output_dim, rngs=rngs))

        self.layers = nnx.Sequential(*layers)

    def __call__(self, x: Array):
        return self.layers(x)


def _load_predefined_net(
    network_version: str, in_dim: int, output_dim: int, rngs: nnx.Rngs
) -> Net:
    if network_version not in PREDEFINED_NETWORKS:
        raise NameError(
            f"Network version {network_version} not in {PREDEFINED_NETWORKS}."
        )

    hidden_layers: tuple[int, ...]
    residual_mid_feature_sizes: tuple[int, ...]
    if network_version == "layers_1024_512_256":
        hidden_layers = (1024, 512, 256)
        residual_mid_feature_sizes = ()
    elif network_version == "layers_512_512_residual_0_128":
        hidden_layers = (512, 512)
        residual_mid_feature_sizes = (0, 128)
    elif network_version == "layers_512_256_128_residual_0_64_32":
        hidden_layers = (512, 256, 128)
        residual_mid_feature_sizes = (0, 64, 32)
    elif network_version == "layers_512_256_256_residual_0_128_128":
        hidden_layers = (512, 256, 256)
        residual_mid_feature_sizes = (0, 128, 128)

    policy_net: Net = Net(
        in_dim,
        hidden_layers,
        output_dim,
        nnx.relu,
        residual_mid_feature_sizes,
        rngs,
    )
    return policy_net
