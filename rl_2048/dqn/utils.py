import math
from collections.abc import Sequence


def flat_one_hot(values: Sequence[int], k: int) -> Sequence[float]:
    locations = (0 if v == 0 else int(math.log2(v)) for v in values)
    return [1.0 if i == loc else 0.0 for loc in locations for i in range(k)]
