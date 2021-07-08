from masuite.environments import Environment
from masuite.logging import terminal_logging

def run(
    alg,
    env: Environment,
    logger,
    num_epochs: int,
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
    
    should_render = hasattr(env, 'render')
    shared_state = env.shared_state
    
    obs = env.reset()
    for epoch in range(num_epochs):
        # only render first episode of each epoch
        finished_rendering_this_epoch = False
        count = 0
        while True:
            if not finished_rendering_this_epoch and should_render:
                env.render()
            
            if hasattr(alg, 'buffers'):
                for idx, buffer in enumerate(alg.buffers):
                    if shared_state:
                        buffer.append_obs(obs)
                    else:
                        buffer.append_obs(obs[idx])
            
            # get actions from agent(s)
            if shared_state:
                acts = [agent.select_action(obs) for agent in agents]
            else:
                acts = [agents[i].select_action(obs[i])
                    for i in range(env.n_players)]
            # print(acts[0])
            
            # send action(s) to env and get new timestep info
            obs, rews, done, _ = env.step(acts)
            alg.track_timestep(acts, rews)
            count += 1
            if done:
                finished_rendering_this_epoch = True
                alg.end_episode()
                obs, done = env.reset(), False
                if alg.batch_over():
                    batch_info = alg.end_epoch()
                    logger.track_epoch(batch_info)
                    print('epoch: %3d \t loss: %.3f \t return: %.3f'%
                    (epoch, batch_info["loss"][0], batch_info["avg_rets"][0]))
                    if logger.checkpoint_due():
                        curr_params = alg.get_agent_params(copy=True)
                        logger.log_checkpoint(curr_params)
                    break

    if should_render and hasattr(env, 'close'):
        env.close()

"""
episode -> the number of steps env.reset() and env returning done = True.
batch -> accumulates multiple episodes (~5000 steps)
epoch -> a single training step
"""
