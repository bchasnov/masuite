from abc import ABCMeta, abstractmethod
# TODO: finish agent abc

class Agent(metaclass=ABCMeta):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def env_dim(self):
        """Defines the shape of the environment observations"""
        pass
    
    @property
    @abstractmethod
    def act_dim(self):
        """Defines the shape of the returned actions"""
        pass

    @abstractmethod
    def select_action(obs):
        """Uses action policy to choose an action given obs (env observations)"""
        pass