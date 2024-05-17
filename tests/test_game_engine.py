from typing import Dict, List, NamedTuple

import pytest

from rl_2048.DQN.replay_memory import Action
from rl_2048.game_engine import GameEngine, MoveResult
from rl_2048.tile import Tile


class GridsActionFixture(NamedTuple):
    grids: List[List[int]]
    action: Action
    expected_move_result: MoveResult


# src: https://stackoverflow.com/questions/8421337/rotating-a-two-dimensional-array-in-python
def rot_grids_90_deg(grids: List[List[int]]) -> List[List[int]]:
    new_grids: List[List[int]] = [[0 for _ in row] for row in grids]
    width: int = len(grids[0])
    for i, row in enumerate(grids):
        for j, g in enumerate(row):
            new_grids[j][width - i - 1] = g

    return new_grids


ACTION_ROTATION_MAP: Dict[Action, Action] = {
    Action.UP: Action.RIGHT,
    Action.RIGHT: Action.DOWN,
    Action.DOWN: Action.LEFT,
    Action.LEFT: Action.UP,
}


def rot_fixture_90_deg(fixture: GridsActionFixture) -> GridsActionFixture:
    grids = rot_grids_90_deg(fixture.grids)
    action: Action = ACTION_ROTATION_MAP[fixture.action]
    return GridsActionFixture(grids, action, fixture.expected_move_result)

@pytest.fixture
def move_up_fixtures() -> List[GridsActionFixture]:
    fixtures: List[GridsActionFixture] = []
    # Move up failure case
    fixtures.append(GridsActionFixture(
        grids = [
            [0, 0, 2, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        action = Action.UP,
        expected_move_result = MoveResult(suc=False, score=0)
    ))
    # Move up successful case
    fixtures.append(GridsActionFixture(
        grids = [
        [0, 0, 0, 4],
        [0, 0, 0, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        ],
        action = Action.UP,
        expected_move_result = MoveResult(suc=True, score=8)
    ))
    return fixtures

def test_game_engine_init():
    tile = Tile(width=4, height=4)
    game_engine = GameEngine(tile)

    assert game_engine.score == 0
    assert not game_engine.game_is_over

def test_game_engine_move(move_up_fixtures: List[GridsActionFixture]):
    tile = Tile(width=4, height=4)
    game_engine = GameEngine(tile)

    for move_up_fixture in move_up_fixtures:
        fixture = move_up_fixture
        # Rotate to test all 4 directions
        for _ in range(len(ACTION_ROTATION_MAP)):
            tile.set_grids(fixture.grids)
            assert game_engine.move(fixture.action) == fixture.expected_move_result
            if fixture.expected_move_result.score > 0:
                assert game_engine.score == fixture.expected_move_result.score
                assert not game_engine.game_is_over
                game_engine.reset()
                assert game_engine.score == 0
                assert not game_engine.game_is_over

            fixture = rot_fixture_90_deg(fixture)
