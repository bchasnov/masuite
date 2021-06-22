from masuite.environments import Environment
from masuite.logging import terminal_logging

def run(
    alg,
    env: Environment,
    logger,
    num_epochs: int,
    verbose: bool=False
)->None:
    """
    Runs an agent on an environment

    Args:
        alg: algorithm instance defining update process between epochs
        env: masuite.Env instance 
        num_episodes: Number of episodes to train for
        verbose: Whether or not to also log to terminal
    """
    agents = alg.agents
    if verbose:
        # TODO: Terminal logging
        pass
    
    should_render = hasattr(env, 'render')
    shared_state = env.shared_state
    
    for _ in range(num_epochs):
        obs = env.reset()
        if hasattr(alg, 'buffer'):
            alg.buffer.append_reset(obs)
        # only render first episode of each epoch
        finished_rendering_this_epoch = False
        while True:
            # if not finished_rendering_this_epoch and should_render:
                # env.render()
                # finished_rendering_this_epoch = True
            
            # get actions from agent(s)
            if shared_state:
                acts = [agent.select_action(obs) for agent in agents]
            else:
                acts = [agents[i].select_action(obs[i])
                    for i in range(env.n_players)]
            
            # send action(s) to env and get new timestep info
            obs, rews, done, _ = env.step(acts)
            batch_info = alg.update(obs, acts, rews, done)
            if batch_info is not None:
                logger.track_epoch(batch_info)
                if logger.checkpoint_due():
                    curr_params = alg.get_agent_params(copy=True)
                    logger.log_checkpoint(curr_params)
                break
            if done:
                obs = env.reset()

    if should_render and hasattr(env, 'close'):
        env.close()