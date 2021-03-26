import gym

def run(agents,
        environment: gym.Env,
        num_episodes: int,
        verbose: bool=False)->None:
    """
    Runs an agent on an environment

    Args:
        agent: The agent to train and evaluate
        environment: The environment to train on
        num_episodes: Number of episodes to train for
        verbose: Whether or not to also log to terminal
    """
    if verbose:
        #TODO: config logger when implemented
        pass
    
    for _ in range(num_epochs):
        obs = env.reset()
        done = False
        while !done: 
            actions = [agent.get_action(obs) for agent in agents]
            obs, rews, done, env_info = env.step(actions)
            algs_info = alg.update(grads)
