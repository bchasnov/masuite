"""Run a policy gradient agent instance on a masuite experiment."""
import masuite
from masuite.algos import experiment
from masuite.algos.pytorch import policy_gradient

def run(CONFIG):
    env = masuite.load_and_record(
        env_name=env_name,
        save_path=CONFIG.save_path,
        logging_mode=CONFIG.logging_mode,
    )

    agents = []
    for i in range(env.n_players):
        agents.append(
        policy_gradient.default_agent(env.observation_space, env.action_space, lr=CONFIG.lr)
        )
    
    experiment.run(
        environment=env,
        num_episodes=CONFIG.num_episodes, 
        verbose=CONFIG.verbose)

def main(CONFIG):
    run(CONFIG)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--save_path', type=float, default=1e-2)
    parser.add_argument('--num_episodes', type=float, default=int(1e3))
    parser.add_argument('--verbose', action='store_true')
  
    CONFIG = parser.parse_args()
    main(CONFIG)
