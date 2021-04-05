import gym
import numpy as np

import masuite.environments.base

class CoupledRiccati:
    def __init__(self,
        game: base.Environment,
        tol: float=1e-20,
        verbose: bool=False):
        """
        Solve the coupled Riccati equations to get each player's gain matrix
        and add them to the class fields
        """
        A = game.A.numpy()
        Bs = game.Bs.numpy()
        Qs = game.Qs.numpy()
        Rs = game.Rs.numpy()

        na = len(Bs)
        nx = A.shape[0]
        nu = [b.shape[1] for b in Bs]
        Ks = [np.zeros((n, nx)) for n in nu]
        Ps = np.copy(Qs)
        _norms = [[0]*na]
        Acl = None
        while True:
            Acl = A - sum(Bs[i] @ Ks[i] for i in range(na))
            Aobs = [A - sum([Bs[j] @ Ks[j] if i!=j else 0
                    for j in range(na)])
                    for i in range(na)]

            Rtot = [sum([Ks[j].T @ Rs[i][j] @ Ks[j][0]
                    for j in range(na)])
                    for i in range(na)]
            
            Ks = [np.linalg.inv(Rs[i][i] + Bs[i].T @ Ps[i] @ Bs[i]) @
                  (Bs[i].T @ Ps[i] @ Aobs[i]) for i in range(na)]

            Ps = [np.solve_discrete_are(Aobs[i], Bs[i], Qs[i] + Rtot[i], Rs[i])
                    for i in range(na)]

            _norms.append(np.linalg.norm(np.array(Ps), axis=(1,2)))
            if np.isclose(sum(_norms[-1]), sum(_norms[-2]), atol=tol, rtol=tol):
                break
        
        Ks = np.array([k for k in Ks])
        Ps = np.array([p for p in Ps])
        if verbose:
            print(10*'=', 'Coupled Riccati Params', 10*'=')
            print(5*'=' + 'Acl' + 5*'=')
            print(Acl)
            print(5*'=' + 'A' + 5*'=')
            print(A)
            print(5*'=' + 'Ks' + 5*'=')
            print(Ks)
            print(5*'=' + 'Ps' + 5*'=')
            print(Ps)

        self.Ks, self.Ps = -Ks, Ps


    def get_policy(self, state):
        """
        Given the current game state, calculate the optimal Riccati policy
        actions
        @param state: current state of the game
        @return torch.FloatTensor: tensor of optimal actions for all players
        """
        # return torch.FloatTensor([-self.Ks[i] @ state for i in range(len(self.Ks))])
        return self.Ks