
##########################
## Natural Gradient for L
##########################
    # X0 = SLA.solve_discrete_are(A-B2 @ L0, B1, Q - L0.T @ R2 @ L0, R1)
    # error[i] = np.trace( (X - X0) @ initial_state )
    # K0 = SLA.inv(R1 + B1.T @ X0 @ B1) @ B1.T @ X0 @ (A-B2 @ L0)
    # error_policy[i] = LA.norm(K0 - K, 'fro') + LA.norm(L0 - L, 'fro')
    # # natural gradient
    # ngd = 2 * ( -R2 @ L0 - B2.T @ X0 @ (A - B1@K0 - B2@L0) )
    # # matrix $O_j$; see paper for definition
    # O1 = R2 - B2.T @ X0 @ B2 + B2.T @ X0 @ B1 @ SLA.inv(R1 + B1.T @ X0 @ B1) @ B1.T @ X0 @ B2
    # # step size
    # stepsize = 0.3/max(abs(LA.eigvals(O1)))
    
    # # update
    # L1 = L0 + stepsize * ngd
    # L0 = L1


##########################
## Natural Gradient for K
##########################
#     W0 = SLA.solve_discrete_are(A-B1 @ K0, B2, -Q - K0.T @ R1 @ K0, R2)
#     X0 = -W0
#     error[i] = np.trace( (X0 - X) @ initial_state )
#     L0 = -SLA.inv(R2 - B2.T @ X0 @ B2) @ B2.T @ X0 @ (A-B1 @ K0)
#     error_policy[i] = LA.norm(L0-L, 'fro') + LA.norm(K0 - K, 'fro')
#     # natural gradient
#     ngd = 2 * ( R1 @ K0 - B1.T @ X0 @ (A-B1@K0 - B2@L0) )
#     # matrix $O_j$; see paper for definition
#     P1 = R1 + B1.T @ X0 @ B1 + B1.T @ X0 @ B2 @ SLA.inv(R2 - B2.T @ X0 @ B2) @ B2.T @ X0 @ B1
#     # step size
#     stepsize = 0.32/max(abs(LA.eigvals(P1)))
    
#     # update
#     K1 = K0 - stepsize * ngd
#     K0 = K1