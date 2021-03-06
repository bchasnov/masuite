from abc import ABCMeta, abstractmethod, abstractproperty

class Environment(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractproperty
    def n_players(self):
        """Defines how many players interact with the env"""
        pass

    @abstractproperty
    def mapping_seed(self, val):
        """Defines the seed env randomization"""
        pass

    @abstractproperty
    def env_dim(self):
        """Defines the dimension(s) of the state/observations"""
        pass

    @abstractmethod
    def step(self, actions):
        """Steps the environment to a new timestep given players' actions"""
        pass

    @abstractmethod
    def reset(self):
        """Resets the environment"""
        raise NotImplementedError


class DiscreteEnvironment(Environment, metaclass=ABCMeta):
    """Base class for an environment with a discrete action space"""
    def __init__(self):
        pass

    @abstractproperty
    def n_acts(self):
        """Defines the dimensions of a single player's inputted actions"""
        pass

    @abstractproperty
    def shared_state(self):
        """Defines whether or not players share a single state"""
        pass
    
    
