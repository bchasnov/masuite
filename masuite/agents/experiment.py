import gym
from masuite.logging import terminal_logging
from masuite.utils.logging import Logging

def run(alg,
        env: gym.Env,
        logger: Logging,
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
        # TODO: Terminal logging
        pass
    
    should_render = hasattr(env, 'render')
    shared_state = env.shared_state
    
    for _ in range(num_epochs):
        obs = env.reset()
        if hasattr(env, 'track'):
            env.track(obs)
        if hasattr(alg, 'buffer'):
            alg.buffer.append_reset(obs)
        done = False
        # render first episode of each epoch
        while True:
            # if not finished_rendering_this_epoch and should_render:
                # env.raw_env.render()
            if shared_state:
                acts = [agent.select_action(obs) for agent in agents]
            else:
                acts = []
                for idx in range(env.n_players):
                    acts.append(agents[idx].select_action(obs[idx]))
            obs, rews, done, env_info = env.step(acts)
            logger.track_env_step(rews, done, env_info)
            batch_info = alg.update(obs, acts, rews, done)
            if batch_info is not None:
                # logger.log_batch_info(batch_info)
                break
            if done:
                obs = env.reset()
    if should_render:
        env.close()