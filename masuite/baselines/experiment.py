import gym

def run(alg,
        agents,
        env: gym.Env,
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
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while done is False: 
            actions = [agent.act(obs) for agent in agents]
            obs, rews, done, env_info = env.step(actions)
            algs_info = alg.step(actions, obs)
            print(algs_info)
