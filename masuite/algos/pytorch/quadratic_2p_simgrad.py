import torch
import torch.autograd

class QuadraticTwoPlayerSimgrad:
    def __init__(env, lrs, agents):
        self.env = env
        self.lrs = lrs
        self.agents = agents

    def step(self, acts, rewards):
        if lrs is None:
            lrs = self.lrs
        grads = []
        for i in range(len(self.optims)):
            grad = autograd.grad(rewards[i], acts[i])
            grads.append(grad)
        
        return grads
    
    def update(self, rews, acts):
        info = dict()
        grads = self..step(acts, rews)
        for i in range(len(agents)):
            self.agents[i].update(agents[i].x-lrs[i]*grads[i])
        
        return info
            