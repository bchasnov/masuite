import gym
from masuite.logging import terminal_logging

def run(alg,
        env: gym.Env,
        num_epochs: int,
        batch_size: int,
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
    
    should_render = hasattr(env.raw_env, 'render')
    
    for i in range(num_epochs):
        obs = env.reset()
        env.track(obs)
        if hasattr(alg, 'buffer'):
            alg.buffer.append_reset(obs)
        done = False
        ep_rews = []
        # render first episode of each epoch
        finished_rendering_this_epoch = False
        # while done is False:
        while True:
            # if not finished_rendering_this_epoch and should_render:
                # env.raw_env.render()
            acts = [agent.select_action(obs) for agent in agents]
            obs, rews, done, env_info = env.step(acts)
            batch_loss, batch_rets, batch_lens = alg.update(obs, acts, rews, done)
            if batch_loss is not None:
                break
            if done:
                obs = env.reset()
    if should_render:
        env.close()