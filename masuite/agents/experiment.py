import gym
from masuite.logging import terminal_logging

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
    agents = alg.agents
    if verbose:
        env = terminal_logging.wrap_environment(env, log_every=True)
    
    for _ in range(num_episodes):
        obs = env.reset()
        if hasattr(alg, buffer):
            alg.buffer.append_reset(obs)
        done = False
        ep_rews = []

        while done is False:
            print('running')
            actions = [agent.act(obs) for agent in agents]
            obs, rews, done, env_info = env.step(actions)
            batch_loss, batch_rets, batch_lens = alg.update(obs, acts, rews, done)
            if batch_loss is not None:
                break
