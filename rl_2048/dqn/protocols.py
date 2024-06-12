from typing import Protocol

from jaxtyping import Array

from rl_2048.dqn.common import Batch, Metrics, PolicyNetOutput


class PolicyNet(Protocol):
    def predict(self, feature: Array) -> PolicyNetOutput: ...

    def optimize(self, batch: Batch) -> Metrics: ...
