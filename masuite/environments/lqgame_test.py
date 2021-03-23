import torch
from lqgame import LinearQuadraticGame
from masuite.experiments.envutils import scalar_zerosum_game

def step_game(game):
    return game.step(torch.FloatTensor([[1.], [1.]]))

def test_step():
    game = scalar_zerosum_game()
    ret = step_game(game)
    assert isinstance(ret[0], torch.Tensor)
    assert isinstance(ret[1], torch.Tensor)
    assert isinstance(ret[2], bool)
    assert isinstance(ret[3], dict)
