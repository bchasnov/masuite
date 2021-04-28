import argparse
import masuite
from masuite import sweep
from masuite.agents import experiment
from masuite.agents.pytorch.policy_gradient.agent import PGAgent
from masuite.environments.cartpole import CartPoleEnv
from masuite.algos.pytorch.simple_pg.simple_pg import SimplePG

parser = argparse.ArgumentParser()
parser.add_argument('--masuite-id', default='cartpole_simplepg/0', type=str,
    help='global flag used to control which environment is loaded')
parser.add_argument('--save_path', default='tmp/masuite', type=str,
    help='where to save masuite results')
parser.add_argument('--logging_mode', default='csv', type=str,
    choices=['csv', 'sqlite', 'terminal'], help='how to log masuite results')
parser.add_argument('--overwrite', default = False, type=bool,
    help='overwrite csv logging file if found')
parser.add_argument('--verbose', default=False, type=bool,
    help='whether or not to use verbose logging to terminal')
parser.add_argument('--num_episodes', default=None, type=int,
    help='overrides number of training episodes')

parser.add_argument('--seed', default=0, type=int,
    help='seed for random number generation')
parser.add_argument('--lr', default=1e-2, type=int,
    help='learning rates for agents')

args = parser.parse_args()

def run(masuite_id: str):
    env = masuite.load_and_record(
        masuite_id=masuite_id,
        save_path=args.save_path,
        logging_mode=args.logging_mode,
        overwrite=args.overwrite
    )

    env_dim = [4]
    n_acts = env.raw_env.action_space.n # number of possible actions
    act_dim = 1 # number of actions chosen at each step (per agent)
    agents = [PGAgent(env_dim=env_dim, n_acts=n_acts, lr=args.lr)]
    alg = SimplePG(
        agents=agents,
        obs_dim=env_dim,
        act_dim=act_dim,
        n_players=1
    )

    num_episodes = args.num_episodes or getattr(env, 'masuite_num_episodes')

    experiment.run(
        alg=alg,
        env=env,
        num_episodes=num_episodes,
        verbose=args.verbose
    )

    return masuite_id


def main():
    masuite_id = args.masuite_id
    if masuite_id in sweep.SWEEP:
        print(f'Running single experiment: masuite_id={masuite_id}')
        run(masuite_id=masuite_id)
    elif hasattr(sweep, masuite_id):
        masuite_sweep = getattr(sweep, masuite_id)
        print(f'Running sweep over masuite_id in sweep.{masuite_id}')
        args.verbose = False
        pool.map_mpi(run, masuite_sweep)
    else:
        raise ValueError(f'Invalid flag: masuite_id={masuite_id}')


if __name__ == '__main__':
    main()