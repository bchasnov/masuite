from abc import ABCMeta, abstractmethod

class Optimizer(metaclass=ABCMeta):
    @abstractmethod
    def step(self):
        """Make a single optimization step"""
