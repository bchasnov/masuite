from abc import ABCMeta, abstractmethod, abstractproperty


class Environment(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractproperty
    def mapping_seed(self, val):
        """Defines the seed env randomization"""
        pass

    @abstractproperty
    def n_players(self):
        """Defines how many players interact with the env"""
        pass

    @abstractproperty
    def env_dim(self):
        """Defines the dimension(s) of the state/observations"""
        pass

    @abstractproperty
    def act_dim(self):
        """Defines the dimensions of a single player's inputted actions"""
        pass

    @abstractproperty
    def shared_state(self):
        """Defines whether or not players share a single state"""
        pass
    
    @abstractmethod
    def step(self, actions):
        """Steps the environment to a new timestep given players' actions"""
        pass
    
    @abstractmethod
    def reset(self):
        """Resets the environment"""
        raise NotImplementedError