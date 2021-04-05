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
    if verbose:
        env = terminal_logging.wrap_environment(env, log_every=True)
    
    for _ in range(num_episodes):
        batch_obs = []     # for observations
        batch_acts = []    # for actions
        batch_weights = [] # for R(taut) weighting in policy gradient
        batch_rets = []    # for measuring episode returns
        batch_lens = []    # for measuring episode lengths

        obs = env.reset()
        done = False
        ep_rews = []

        while done is False: 
            batch_obs.append(obs.copy())
            actions = [agent.act(obs) for agent in agents]
            obs, rews, done, env_info = env.step(actions)
            batch_acts.append(act)
            ep_rews.append(rew)
            if done:
                #TODO: convert for multi-dim arrays
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                #TODO: maybe move weights to simple pg
                batch_weights += [ep_ret] * ep_len

            updates = alg.step(batch_acts, batch_obs, weights)
            agents[i].update(obs, actions, updates[i]) for i in range(len(agents))

