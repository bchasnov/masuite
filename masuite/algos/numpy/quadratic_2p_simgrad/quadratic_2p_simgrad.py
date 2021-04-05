import numpy as np

class QuadraticTwoPlayerSimgrad:
    def __init__(self, agents, env, lrs):
        self.agents = agents
        self.env = env
        self.lrs = lrs

    def step(self, acts, obs, lrs=None):
        info = {}
        if lrs is None:
            lrs = self.lrs
        A, B, C, D = self.env.A, self.env.B, self.env.C, self.env.D
        x, y = acts[0], acts[1]
        g1 = A@x + B@y
        g2 = C@x + D@y
        return (g1, g2)
        self.agents[0].update(x-lrs[0]*g1)
        self.agents[1].update(y-lrs[1]*g2)
        info['agent 1 update'] = x-lrs[0]*g1
        info['agent 2 update'] = x-lrs[1]*g2
        return (x-lrs[0]*g1, y-lrs[1]*g2)
        