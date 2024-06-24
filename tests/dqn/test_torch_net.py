import pytest
import torch
import torch.nn as nn

from rl_2048.dqn.common import (
    PREDEFINED_NETWORKS,
    Batch,
    TrainingParameters,
)
from rl_2048.dqn.torch_net import Net, TorchPolicyNet


def test_net():
    torch.manual_seed(0)

    net1 = Net(16, 4, [64, 16], nn.ReLU(), [])
    net1.eval()
    input_tensor = torch.rand([1, 16])
    assert net1(input_tensor).shape == (1, 4)

    net2 = Net(16, 4, [32, 32], nn.ReLU(), [0, 16])
    net2.eval()
    assert net2(input_tensor).shape == (1, 4)

    net3 = Net(16, 4, [32, 16], nn.ReLU(), [0, 8])
    net3.eval()
    assert net3(input_tensor).shape == (1, 4)


def test_net_errors():
    with pytest.raises(ValueError):
        Net(16, 4, [64, 16], nn.ReLU(), [1, 1, 1])

    with pytest.raises(ValueError):
        Net(16, 4, [32, 30], nn.ReLU(), [0, 16])


@pytest.fixture
def batch_size() -> int:
    return 10


@pytest.fixture
def feature_size() -> int:
    return 16


@pytest.fixture
def batch(batch_size: int, feature_size: int):
    return Batch(
        states=torch.rand((batch_size, feature_size)).tolist(),
        actions=torch.randint(0, 4, (batch_size,)).tolist(),
        next_states=torch.rand((batch_size, feature_size)).tolist(),
        rewards=torch.rand((batch_size,)).tolist(),
        games_over=torch.randint(0, 2, (batch_size,), dtype=torch.bool).tolist(),
    )


def test_policy_net_predefined(batch):
    torch.manual_seed(0)

    for network_name in PREDEFINED_NETWORKS:
        training_parameters = TrainingParameters(loss_fn="HuberLoss")
        policy_net = TorchPolicyNet(network_name, 16, 4, training_parameters)
        policy_net.optimize(batch)
        _ = policy_net.predict(torch.rand((16,)).tolist())


def test_policy_net_parameters_same_lr_gamma(batch):
    training_parameters = TrainingParameters(
        loss_fn="HuberLoss", lr=0.1, lr_decay_milestones=[1, 2], lr_gamma=0.1
    )
    policy_net = TorchPolicyNet("layers_1024_512_256", 16, 4, training_parameters)
    metrics1 = policy_net.optimize(batch)
    assert metrics1["step"] == 1
    assert metrics1["lr"] == pytest.approx(0.1)
    metrics2 = policy_net.optimize(batch)
    assert metrics2["step"] == 2
    assert metrics2["lr"] == pytest.approx(0.01)
    metrics3 = policy_net.optimize(batch)
    assert metrics3["step"] == 3
    assert metrics3["lr"] == pytest.approx(0.001)


def test_policy_net_parameters_different_lr_gamma(batch):
    training_parameters = TrainingParameters(
        loss_fn="HuberLoss", lr=0.1, lr_decay_milestones=[1, 2], lr_gamma=[0.5, 0.1]
    )
    policy_net = TorchPolicyNet("layers_1024_512_256", 16, 4, training_parameters)
    metrics1 = policy_net.optimize(batch)
    assert metrics1["step"] == 1
    assert metrics1["lr"] == pytest.approx(0.1)
    metrics2 = policy_net.optimize(batch)
    assert metrics2["step"] == 2
    assert metrics2["lr"] == pytest.approx(0.05)
    metrics3 = policy_net.optimize(batch)
    assert metrics3["step"] == 3
    assert metrics3["lr"] == pytest.approx(0.005)


def test_policy_net_parameters_constant_lr(batch):
    training_parameters = TrainingParameters(
        loss_fn="HuberLoss", lr=0.1, lr_decay_milestones=[], lr_gamma=[]
    )
    policy_net = TorchPolicyNet("layers_1024_512_256", 16, 4, training_parameters)
    for i in range(6):
        metrics = policy_net.optimize(batch)
        assert metrics["step"] == i + 1
        assert metrics["lr"] == pytest.approx(0.1)


def test_policy_net_errors():
    with pytest.raises(NameError):
        training_parameters = TrainingParameters(loss_fn="HuberLoss")
        TorchPolicyNet("random name", 16, 4, training_parameters)

    with pytest.raises(ValueError):
        training_parameters = TrainingParameters(
            loss_fn="HuberLoss", lr_decay_milestones=10, lr_gamma=[0.1, 0.1]
        )
        TorchPolicyNet("layers_1024_512_256", 16, 4, training_parameters)

    with pytest.raises(ValueError):
        training_parameters = TrainingParameters(
            loss_fn="HuberLoss", lr_decay_milestones=[1], lr_gamma=[0.1, 0.1]
        )
        TorchPolicyNet("layers_1024_512_256", 16, 4, training_parameters)
