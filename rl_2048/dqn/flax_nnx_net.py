"""
Implement the following protocol

class PolicyNet(Protocol):
    def predict(self, feature: Sequence[float]) -> PolicyNetOutput: ...

    def optimize(self, batch: Batch) -> Metrics: ...

    def save(self, filename_prefix: str) -> str: ...

    def load(self, model_path: str): ...
"""

from typing import Callable

from flax import nnx
from jaxtyping import Array


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
