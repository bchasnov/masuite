from typing import Iterable
from torch import Tensor
from masuite.agents.pytorch.optims.base import Optimizer

class SGDOptim(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float):
        self.params = params 
        self.lr = lr
    
    
    def step(self):
        for param in self.params:
            param.data.add_(-self.lr * param.grad)
