


# Run the seed scanning experiment to find a setting where
# lqr simgrad fails but both lqr stack and lqr tau-simgrad succeeds

# 

# to run and collect trajectories, use the zerosum simgrad and zerosum dare algos in numpy

# plot analysis afterwards

L, K # from algos.numpy.lqgame_zs_dare

for seed in range(start_seed, int(1e8)):
    np.random.seed(seed)
    L0 = L + delta*np.random.randn(*L.shape)
    K0 = K + delta*np.random.randn(*K.shape)
    for i in range(MAX_ITER):
        # iterate from  algos.numpy.lqgame_zs_simgrad


# seed found

# run algos.numpy.lqgame_zs_simgrad with tau separation
# run algos.numpy.lqgame_zs_stack 
# from both sides
# from both sides