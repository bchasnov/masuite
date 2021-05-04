import gym
from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def mapping_seed(self, val):
        pass

    @property
    @abstractmethod
    def n_players(self):
        pass

    @property
    @abstractmethod
    def env_dim(self):
        pass

    @property
    @abstractmethod
    def act_dim(self):
        pass

    @property
    @abstractmethod
    def shared_state(self):
        pass
    
    @property
    @abstractmethod
    def step(self, actions):
        pass
    
    @property
    @abstractmethod
    def reset(self):
        raise NotImplementedError