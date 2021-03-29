import gym

class Environment(gym.Env):
    def __init__(self):
        self.n_players = None

    def step(self, actions):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError