import gym
from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def mapping_seed(self, val):
        """Defines the seed env randomization"""
        pass

    @property
    @abstractmethod
    def n_players(self):
        """Defines how many players interact with the env"""
        pass

    @property
    @abstractmethod
    def env_dim(self):
        """Defines the dimension(s) of the state/observations"""
        pass

    @property
    @abstractmethod
    def act_dim(self):
        """Defines the dimensions of a single player's inputted actions"""
        pass

    @property
    @abstractmethod
    def shared_state(self):
        """Defines whether or not players share a single state"""
        pass
    
    @property
    @abstractmethod
    def step(self, actions):
        """Steps the environment to a new timestep given players' actions"""
        pass
    
    @property
    @abstractmethod
    def reset(self):
        """Resets the environment"""
        raise NotImplementedError