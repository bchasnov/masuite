"""Run a policy gradient agent instance on a masuite experiment."""
from masuite.baselines import experiment
from masuite.baselines.pytorch import policy_gradient
from masuite.utils import dict2config

def run(CONFIG):
  env = masuite.load_and_record(
    env_id=env_id,
    save_path=CONFIG.save_path,
    logging_mode=CONFIG.logging_mode,
  )

  agents = []
  for i in range(env.n_players):
    agents.append(
      policy_gradient.default_agent(env.observation_spec(), env.action_spec())
    )
  
  alg = masuite.algorithm(
    alg_id=alg_id,
    agents=agents,
    save_path=CONFIG.save_path,
    logging_mode=CONFIG.logging_mode,
  )
  
  experiment.run(
    agents=agents,
    environment=env,
    algorithm=alg,
    num_episodes=CONFIG.num_episodes, 
    verbose=CONFIG.verbose)


def main():
  run(CONFIG)

if __name__ == '__main__':
  CONFIG = dict2config(env_id='lqgame-2p-zs', alg_id='simgrad', save_path='', num_episodes=int(1e4), verbose=true)
  main()
