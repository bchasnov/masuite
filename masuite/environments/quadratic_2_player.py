from masuite.environments.base import Environment
import numpy as np

class QuadraticTwoPlayer(Environment):
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray):
        self.n_players = 2
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    
    def step(self, actions):
        """
        Take a single step given actions, and return player 1 and player 2 reward
        for actions

        parameters:
            actions: np.ndarray of actions
        """
        assert len(actions) == 2, 'Actions must be np.ndarray of length 2'
        x, y = actions[0], actions[1]
        
        return None, (x.T@self.A@x/2 + y.T*self.B*x, y.T*self.D*y/2 + x.T*self.C*y), True, {}
    
    def reset(self):
        return None
