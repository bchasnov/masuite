

class LinearQuadraticGameZerosumSimgrad:

  def __init__(self, env, agents):
    self.env = env
    self.agents = agents
    # TODO: add environment variables

  def step(self, actions):
    info = dict()
    K0, L0 = actions

    try:
      X0 = SLA.solve_discrete_are(A-B2 @ L0, B1, Q - L0.T @ R2 @ L0, R1)
      W0 = SLA.solve_discrete_are(A-B1 @ K0, B2, -Q - K0.T @ R1 @ K0, R2)
    except LA.LinAlgError:
      #print('Failed to solve DARE at iteration', i ,')\n', 'sp(Acl)=', specrad[i])
      info['RAISE ERROR'] = 'LinAlgError'

    gd_k = 2 * ( R1 @ K0 - B1.T @ (-W0) @ (A-B1@K0 - B2@L0) )
    gd_l = 2 * ( -R2 @ L0 - B2.T @ X0 @ (A-B1@K0 - B2@L0) )
    
    
    K1 = K0 - lr1 * gd_k
    L1 = L0 + lr2 * gd_l

    agents[0].update(K1)
    agents[1].update(L1)

    error1 = np.trace( (X - X0) @ initial_state )
    error2 = np.trace( (-X - W0) @ initial_state )

    info['error'] = (error1, error2)
    info['grads'] = (gd_k, gd_l)
    info['profiles'] = (K1, L1)

    return info