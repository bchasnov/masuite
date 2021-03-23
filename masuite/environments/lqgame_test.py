import torch
from lqgame import LinearQuadraticGame
from masuite.utils.envutils import scalar_zerosum_game

def step_game(**params):
    game = LinearQuadraticGame(**params)
    return game.step(torch.FloatTensor([[1.]]))

def test_step():
    params = scalar_zerosum_game()
    assert step_game(**params) == 0