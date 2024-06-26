import tempfile

import pytest
from jax import Array
from jax import random as jrandom

import rl_2048.dqn as common_dqn
from rl_2048.dqn.common import (
    PREDEFINED_NETWORKS,
    Action,
    DQNParameters,
    TrainingParameters,
)
from rl_2048.dqn.jax_net import (
    JaxPolicyNet,
)
from rl_2048.dqn.replay_memory import Transition

# def test_dqn():
#     input_dim = 100
#     output_dim = 4
#     dqn_params = DQNParameters(memory_capacity=4, batch_size=2)
#     training_params = TrainingParameters(
#         gamma=0.99,
#         batch_size=2,
#         lr=0.001,
#         eps_start=0.0,
#         eps_end=0.0,
#     )
#     rng: Array = jrandom.key(0)
#     t1 = Transition(
#         state=jrandom.normal(rng, shape=(input_dim,)).tolist(),
#         action=Action.UP,
#         next_state=jrandom.normal(rng, shape=(input_dim,)).tolist(),
#         reward=10.0,
#         game_over=False,
#     )
#     t2 = Transition(
#         state=jrandom.normal(rng, shape=(input_dim,)).tolist(),
#         action=Action.LEFT,
#         next_state=jrandom.normal(rng, shape=(input_dim,)).tolist(),
#         reward=-1.0,
#         game_over=False,
#     )

#     for network_version in PREDEFINED_NETWORKS:
#         policy_net: Net = _load_predefined_net(network_version, output_dim)
#         policy_net.check_correctness()

#         with tempfile.TemporaryDirectory() as tmp_dir:
#             dqn = DQN(input_dim, policy_net, tmp_dir, dqn_params, training_params, rng)

#             dqn.push_transition(t1)
#             dqn.push_transition(t2)
#             loss = dqn.optimize_model(0)
#             assert loss != 0.0

#             print(dqn.get_action_epsilon_greedy(t2.state))

#             model_path = dqn.save_model()
#             dqn.load_model(model_path)


# def test_learning_rate_fn_int_float():
#     params = TrainingParameters(lr=0.1, lr_decay_milestones=5, lr_gamma=0.1)
#     lr_fn: Schedule = _create_lr_scheduler(params)
#     lrs: list[float] = [lr_fn(i) for i in range(15)]
#     expected_lrs_int_float: list[float] = [0.1] * 5 + [0.01] * 5 + [0.001] * 5
#     assert lrs == pytest.approx(expected_lrs_int_float, rel=1e-6)


# def test_learning_rate_fn_int_listoffloat():
#     params = TrainingParameters(lr=0.1, lr_decay_milestones=5, lr_gamma=[0.1, 0.1])
#     with pytest.raises(ValueError):
#         _create_lr_scheduler(params)


# def test_learning_rate_fn_listofint_float():
#     params = TrainingParameters(lr=0.1, lr_decay_milestones=[3, 6], lr_gamma=0.1)
#     lr_fn: Schedule = _create_lr_scheduler(params)
#     lrs: list[float] = [lr_fn(i) for i in range(10)]
#     expected_lrs_int_float: list[float] = [0.1] * 3 + [0.01] * 3 + [0.001] * 4
#     assert lrs == pytest.approx(expected_lrs_int_float, rel=1e-6)


# def test_learning_rate_fn_listofint_listoffloat():
#     params = TrainingParameters(lr=0.1, lr_decay_milestones=[3, 6], lr_gamma=[0.5, 0.1])
#     lr_fn: Schedule = _create_lr_scheduler(params)
#     lrs: list[float] = [lr_fn(i) for i in range(10)]
#     expected_lrs_int_float: list[float] = [0.1] * 3 + [0.05] * 3 + [0.005] * 4
#     assert lrs == pytest.approx(expected_lrs_int_float, rel=1e-6)


# def test_learning_rate_fn_listofint_listoffloat_gt0():
#     params = TrainingParameters(lr=0.1, lr_decay_milestones=[3, 6], lr_gamma=[0.5, 2.0])
#     lr_fn: Schedule = _create_lr_scheduler(params)
#     lrs: list[float] = [lr_fn(i) for i in range(10)]
#     expected_lrs_int_float: list[float] = [0.1] * 3 + [0.05] * 3 + [0.1] * 4
#     assert lrs == pytest.approx(expected_lrs_int_float, rel=1e-6)


# def test_train_step_lr():
#     params = TrainingParameters(lr=0.1, lr_decay_milestones=[3, 6], lr_gamma=[0.1, 5.0])
#     lr_fn: Schedule = _create_lr_scheduler(params)

#     rng: Array = jrandom.key(0)
#     net: nn.Module = Net((2,), 4, nn.relu, (0,))
#     input_dim: int = 2
#     optimizer_str: str = "adamw"
#     loss_fn_str: str = "huber_loss"
#     loss_fn = getattr(optax, loss_fn_str)
#     train_state = create_train_state(
#         rng,
#         net,
#         input_dim,
#         optimizer_str,
#         lr_fn,
#     )
#     batch = JaxBatch(
#         states=jrandom.uniform(rng, (4, input_dim)),
#         actions=jrandom.randint(rng, (4, 1), 0, 4),
#         next_states=jrandom.uniform(rng, (4, input_dim)),
#         rewards=jrandom.uniform(rng, (4, 1)),
#         games_over=jrandom.randint(rng, (4, 1), 0, 2),
#     )
#     for _ in range(10):
#         train_state, _loss, _step, lr = train_step(
#             train_state,
#             batch,
#             jrandom.uniform(rng, (4, input_dim)),
#             lr_fn,
#             optax_loss_fn=loss_fn,
#         )
#         i = train_state.step  # step begins with 1, not 0
#         expected: float = 0.1 if i <= 3 else (0.01 if i <= 6 else 0.05)
#         assert lr == pytest.approx(expected, rel=1e-6), f"i: {i}, lr: {lr}"


def test_jax_policy_net():
    input_dim = 100
    output_dim = 4
    dqn_params = DQNParameters(
        memory_capacity=4,
        batch_size=2,
        eps_start=0.0,
        eps_end=0.0,
    )
    training_params = TrainingParameters(
        gamma=0.99,
        lr=0.001,
    )
    rng: Array = jrandom.key(0)
    t1 = Transition(
        state=jrandom.normal(rng, shape=(input_dim,)).tolist(),
        action=Action.UP,
        next_state=jrandom.normal(rng, shape=(input_dim,)).tolist(),
        reward=10.0,
        game_over=False,
    )
    t2 = Transition(
        state=jrandom.normal(rng, shape=(input_dim,)).tolist(),
        action=Action.LEFT,
        next_state=jrandom.normal(rng, shape=(input_dim,)).tolist(),
        reward=-1.0,
        game_over=False,
    )

    test_feature = jrandom.normal(rng, shape=(input_dim,)).tolist()

    for network_version in PREDEFINED_NETWORKS:
        policy_net = JaxPolicyNet(
            network_version, input_dim, output_dim, rng, training_params
        )
        policy_net.check_correctness()

        with tempfile.TemporaryDirectory() as tmp_dir:
            dqn = common_dqn.DQN(policy_net, dqn_params)

            dqn.push_transition(t1)
            dqn.push_transition(t2)
            loss = dqn.optimize_model()
            assert loss != 0.0

            _ = dqn.get_action_epsilon_greedy(t2.state)

            model_path = dqn.save_model(tmp_dir)

            policy_net_2 = JaxPolicyNet(network_version, input_dim, output_dim, rng)
            dqn_load_model = common_dqn.DQN(policy_net_2)
            dqn_load_model.load_model(model_path)

            assert dqn_load_model.predict(test_feature).expected_value == pytest.approx(
                dqn.predict(test_feature).expected_value
            )
