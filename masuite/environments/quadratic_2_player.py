import numpy as np

class QuadraticTwoPlayer:
    def __init__(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    
    def step(actions):
        """
        Take a single stack given actions, and return player 1 and player 2 reward
        for actions

        parameters:
            actions: np.ndarray of actions
        """
        assert len(actions) == 2, 'Actions must be np.ndarray of length 2'
        x, y = actions[0], actions[1]
        return None, (x.T@A@x/2 + y.T@B@x, y.T@D@y/2 + x.T@C@y), True, {}
    
    def reset(self):
        return None

