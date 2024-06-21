import tempfile

from rl_2048.dqn.common import (
    Action,
    DQNParameters,
    TrainingParameters,
)
from rl_2048.dqn.torch.dqn import DQN
from rl_2048.dqn.torch.net import TorchPolicyNet
from rl_2048.dqn.torch.replay_memory import Transition


def test_torch_dqn():
    training_parameters = TrainingParameters(
        loss_fn="HuberLoss", lr=0.1, lr_decay_milestones=[1, 2], lr_gamma=0.1
    )
    dqn_parameters = DQNParameters(memory_capacity=10, batch_size=2)
    policy_net = TorchPolicyNet("layers_1024_512_256", 2, 4, training_parameters)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dqn = DQN(policy_net, dqn_parameters, tmp_dir)

        t1 = Transition(
            state=[1.0, 0.5],
            action=Action.UP,
            next_state=[2.0, 0.0],
            reward=10.0,
            game_over=False,
        )
        t2 = Transition(
            state=[2.0, 0.0],
            action=Action.LEFT,
            next_state=[-0.5, 1.0],
            reward=-1.0,
            game_over=False,
        )

        dqn.push_transition(t1)
        dqn.push_transition(t2)
        dqn.optimize_model()

        print(dqn.get_action_epsilon_greedy(t2.state))

        model_path = dqn.save_model(f"{tmp_dir}/model")
        dqn.load_model(model_path)
