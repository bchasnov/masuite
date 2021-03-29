

class LinearQuadraticGameZerosumDare:
  def __init__(self)

    pass 

  def step(self, actions=None):
    dim1 = 2
    X = SLA.solve_discrete_are(A, B, Q, R)
    K_opt = LA.inv( R + B.T @ X @ B ) @ B.T @ X @ A
    L, K = K_opt[:dim1, :], K_opt[dim1:, :]



    agents.update(L)
    agents.update(K)
    # TODO: check if it is actually -X and signs of K,L
    info = dict(P1=X, P2=-X, K1=K, K2=L)

    return info