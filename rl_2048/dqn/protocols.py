from collections.abc import Sequence
from typing import Protocol

from rl_2048.dqn.common import Batch, Metrics, PolicyNetOutput


class PolicyNet(Protocol):
    def predict(self, feature: Sequence[float]) -> PolicyNetOutput: ...

    def optimize(self, batch: Batch) -> Metrics: ...

    def save(self, filename_prefix: str) -> str: ...

    def load(self, model_path: str): ...
