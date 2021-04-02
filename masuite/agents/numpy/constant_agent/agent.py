import numpy as np

class ConstantAgent:
    def __init__(self, env_dim, act_dim):
        self.x = np.zeros(act_dim)
    
    def act(self, obs):
        return self.x

    def update(self, state):
        self.x = state
