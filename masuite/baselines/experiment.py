import gym

def run(agent,
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
    
    for _ in range()