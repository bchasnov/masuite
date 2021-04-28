from masuite.environments.base import Environment
import numpy as np

class GridEnv(Environment):
    def __init__(self, mapping_seed, n_players=2):
        self.ball_possesion = np.random.randint(1, n_players+1)


class GridPlayer():
    def __init__(self, init_state=None):

