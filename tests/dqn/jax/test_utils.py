from collections.abc import Sequence

from rl_2048.dqn.jax.utils import flat_one_hot


def test_flat_one_hot():
    input: Sequence[int] = [0, 2, 4, 8, 16]
    features: Sequence[float] = flat_one_hot(input, 5)
    # fmt: off
    expected: Sequence[float] = [
        1., 0., 0., 0., 0.,
        0., 1., 0., 0., 0.,
        0., 0., 1., 0., 0.,
        0., 0., 0., 1., 0.,
        0., 0., 0., 0., 1.,
    ]
    # fmt: on
    assert len(features) == len(expected) and all(
        a == b for a, b in zip(features, expected)
    )
