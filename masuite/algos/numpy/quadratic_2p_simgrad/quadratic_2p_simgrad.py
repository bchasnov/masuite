import numpy as np

class QuadraticTwoPlayerSimgrad:
    def __init__(env, lrs, agents):
        self.env = env
        self.agents = agents
        self.lrs = lrs

    def step(self, acts, obs, lrs=None):
        info = {}
        if lrs is None:
            lrs = self.lrs
        A, B, C, D = env.A, env.B, env.C, env.D
        x, y = acts[0], acts[1]
        g1 = A@x + B@y
        g2 = C@x + D@y
        self.agents[0].update(x-lrs[0]*g1)
        self.agents[1].update(y-lrs[1]*g2)
        return info
        