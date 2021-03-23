import torch
from gym.spaces import Box, Discrete

"""
Contains various functions to creat Linear Quadratic systems for use in the
LinearQuadraticGame class. All functions must return:
    - state transition matrix (A)
    - control input matrices (Bs)
    - reward transition matrices (Qs)
    - player action weights (Rs)
    - intial state of the system (x0)
All which of are type torch.FloatTensor

Functions can optionally return:
    - list of maximum values states can take on (high)
    - list of minimum values states can take on (low)
which are torch.FloatTensors, and:
    - Space object corresponding to valid actions (action_space)
    - Space object corresponding to valid observations (observation_space)
which are of type gym.spaces.Space
"""


def aircraft():
    """
    system that apparently corresponds to a fighter aircraft
    """

    A = torch.FloatTensor([[0.956488,  0.0816012, -0.0005],
                           [0.0741349, 0.94121,   -0.000708383],
                           [0,         0,         0.132655]])

    Bs = torch.FloatTensor([[[-0.00550808],
                             [-0.096],
                             [0.867345]],
                             
                            [[0.00951892],
                             [0.0038733],
                             [0.001]]])
    
    Rs = torch.FloatTensor([[[[1.]], [[0]]], [[[0]], [[1.]]]])

    Qs = torch.FloatTensor(
        [[[1., 0., 0],
          [0., 1., 0],
          [0., 0., 1.]],
          
         [[-1., 0., 0.],
          [0., -1., 0.],
          [0., 0., -1.]]]
    )

    # x0 = torch.FloatTensor([[1.],
    #                         [2.],
    #                         [3.]])
    x0 = torch.randn((A.shape[0], Bs.shape[2]))

    return A, Bs, Qs, Rs, x0


def simple_system():
    """
    simple system that is stable
    """
    A = torch.FloatTensor([[0.5, 0.0, 0.0],
                           [0.0, 0.5, 0.0],
                           [0.0, 0.0, 0.5]])
    
    Bs = torch.FloatTensor([[[1.0],
                             [1.0],
                             [1.0]],
                             
                            [[1.0],
                             [1.0],
                             [1.0]]])
    
    Qs = torch.FloatTensor([[[1., 0., 0],
                             [0., 1., 0],
                             [0., 0., 1.]],
          
                            [[-1., 0., 0.],
                             [0., -1., 0.],
                             [0., 0., -1.]]])
    
    Rs = torch.FloatTensor([[[[10]], [[-10]]], [[[-10]], [[10]]]])

    x0 = torch.randn((3))
    
    return A, Bs, Qs, Rs, x0


def scalar_zerosum(a=0.9, b1=.2, b2=.05, q=10, r1=1, r2=-.5):
    import numpy as np
    A = torch.FloatTensor([[a]])
    Bs = torch.FloatTensor([[[b1]], [[b2]]])
    Q = np.array([[q]])
    R = np.array([[r1], [r2]])
    Qs = torch.FloatTensor([Q, -Q])
    Rs = torch.FloatTensor([R, -R])
    x0 = torch.FloatTensor([[1.]])
    return A, Bs, Qs, Rs, x0

def scalar_cooperative(a=0.9, b1=.1, b2=.1, q=1, r1=2, r2=3):
    import numpy as np
    A = torch.FloatTensor([[a]])
    Bs = torch.FloatTensor([[[b1]], [[b2]]])
    Q = np.array([[q]])
    R = np.array([[r1], [r2]])
    Qs = torch.FloatTensor([Q, Q])
    Rs = torch.FloatTensor([R, R])
    x0 = torch.FloatTensor([[0.]])
    return A, Bs, Qs, Rs, x0

def system1():
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
                                [1],  # p2 location
                                [0]]) # p2 velocity

    Qs = torch.FloatTensor([[[1,0,0,0],
                            [0,0,0,0],
                            [0,0,.5,0],
                            [0,0,0,0]],
                            
                            [[.5,0,0,0],
                            [0,0,0,0],
                            [0,0,1,0],
                            [0,0,0,0]]])

    Qs = torch.FloatTensor([[[1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]],
                            
                            [[1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]]])

    Rs = torch.FloatTensor([[[[1.0]],
                            [[-1.0]]],
        
                            [[[-1.0]],
                            [[1.0]]]])
    
    x0 = x0 = torch.FloatTensor([[1.0],
                            [1.0],
                            [1.0],
                            [1.0]])

    return A, Bs, Qs, Rs, x0


