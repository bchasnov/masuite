import numpy as np
from gym import spaces
import masuite.utils.systems as systems
from masuite.environments.lqgame import LinearQuadraticGame

def create_game_from_matrices(A, Bs, Qs, Rs, x0, obs_space_max=500):
    """
    Create a LinearQuadraticGame instance using the matrices provided as
    function arguments.

    @param A np.ndarray: state transition matrix
    @param Bs np.ndarray: control input matrices (1 for each player)
    @param Qs np.ndarray: reward transition matrix
    @param Rs np.ndarray: reward weights for each player. Must have the shape
    |n_players|x|n_players|
    @param obs_space_max int: maximum valid value states can reach

    @return: LinearQuadraticGame
    """
    vals = np.ones((x0.shape[0], 1)) * obs_space_max
    observation_space = spaces.Box(-vals, vals)
    game = LinearQuadraticGame(
        n_players=Rs.shape[0],
        n_states=x0.shape[0],
        n_actions=1 if np.isscalar(Rs[0][0]) else len(Rs[0][0]),
        A=A,
        Bs=Bs,
        Qs=Qs,
        Rs=Rs,
        states=x0,
        observation_space=observation_space
    )

    return game

def create_game_from_system(system, obs_space_max=500):
    """
    Create a LinearQuadraticGame instance using the system dynamics returned
    by the system parameter.

    @param system function that defines the game dynamics and returns the
    matrices A, Bs, Qs, Rs, and x0 all of type np.array
    """
    A, Bs, Qs, Rs, x0 = system()
    return create_game_from_matrices(A, Bs, Qs, Rs, x0, obs_space_max)

    return game


def aircraft_game():
    """
    Create a LQ game environment with the fighter aircraft simulation dynamics

    @return LinearQuadraticGame
    """
    A, Bs, Qs, Rs, x0 = systems.aircraft()

    return create_game_from_matrices(A, Bs, Qs, Rs, x0)


def simple_game():
    return create_game_from_matrices(*systems.simple_system())

def scalar_zerosum_game():
    return create_game_from_matrices(*systems.scalar_zerosum())

def scalar_cooperative_game():
    return create_game_from_matrices(*systems.scalar_cooperative())

def vpg_game(game=None):
    """
    Return an LQ game for use with VPG algorithm. Note: this functions should
    be passed to vpg as the \"env_fn\" argument.

    @param params: dictionary with all parameters for a new LQ game
    """
    if game is not None: return game

    dt = 0.01

    A = torch.FloatTensor([[1, dt, 0, 0],
                           [0, 1,  dt, 0],
                           [0, 0,  1, dt],
                           [0, 0,  0, 1]])
    
    Bs = torch.FloatTensor([[[0.0],
                         [0.0],
                         [0.0],
                         [1.0]],
                
                        [[0.0],
                         [0.0],
                         [0.0],
                         [1.0]]])

    states = torch.FloatTensor([[1.], # p1 location
                                [0],  # p1 velocity
                                [1.],  # p2 location
                                [0]]) # p2 velocity

    Qs = torch.FloatTensor([[[1,0,0,0],
                             [0,1,0,0],
                             [0,0,1,0],
                             [0,0,0,1]],
                            
                            [[1,0,0,0],
                             [0,1,0,0],
                             [0,0,1,0],
                             [0,0,0,1]]])

    Rs = torch.FloatTensor([[[1.0],
                             [-1.0]],
        
                            [[-1.0],
                             [1.0]]])

    game = LinearQuadraticGame(
        n_players=2,
        n_states=2,
        n_actions=1,
        A=A,
        Bs=Bs,
        Qs=Qs,
        Rs=Rs,
        states=states
    )

    return game
