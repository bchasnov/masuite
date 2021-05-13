import argparse
import masuite
from masuite import sweep
from masuite.agents import experiment
from masuite.agents.pytorch.policy_gradient.agent import PGAgent
from masuite.environments.cartpole import CartPoleEnv
from masuite.algos.pytorch.simple_pg.simple_pg import SimplePG

parser = argparse.ArgumentParser()
# masuite logging and env params
parser.add_argument('--masuite-id', default='cartpole_simplepg/0', type=str,
    help='global flag used to control which environment is loaded')
parser.add_argument('--save_path', default='tmp/masuite', type=str,
    help='where to save masuite results')
parser.add_argument('--logging-mode', default='csv', type=str,
    choices=['csv', 'terminal'], help='how to log masuite results')
parser.add_argument('--overwrite', default = False, type=bool,
    help='overwrite csv logging file if found')
parser.add_argument('--log-by-step', default=False, type=bool,
    help='whether to log by steps rather than on episode completion')
parser.add_argument('--log-every', default=False, type=bool,
    help='whether or not to log every single environment timestep')
parser.add_argument('--log-freq', default=10, type=int,
    help='frequency at which to log env info')
parser.add_argument('--verbose', default=False, type=bool,
    help='whether or not to use verbose logging to terminal')

# algorithm-specific params
parser.add_argument('--num_epochs', default=50, type=int,
    help='number of training epochs')
parser.add_argument('--batch_size', default=5000, type=int,
    help='maximum training batch size per epoch')
parser.add_argument('--seed', default=0, type=int,
    help='seed for random number generation')
parser.add_argument('--lr', default=1e-2, type=int,
    help='learning rate for agents')

args = parser.parse_args()

def run(masuite_id: str):
    env = masuite.load_from_id(masuite_id)

    n_players = env.n_players # number of players
    env_dim = env.env_dim # shape of env state/observations
    n_acts = env.action_space.n # number of possible actions
    act_dim = env.act_dim # number of actions chosen at each step (per agent)
    shared_state = env.shared_state # whether or not all players see the same state

    logger = masuite.init_logging(
        masuite_id=masuite_id,
        n_players=n_players,
        mode=args.logging_mode,
        save_path=args.save_path,
        overwrite=args.overwrite,
        log_by_step=args.log_by_step,
        log_every=args.log_every,
        log_freq=args.log_freq
    )

    agents = [PGAgent(env_dim=env_dim, n_acts=n_acts, lr=args.lr)
        for _ in range(env.n_players)]
    
    alg = SimplePG(
        agents=agents,
        obs_dim=env_dim,
        act_dim=act_dim,
        shared_state=shared_state,
        n_players=env.n_players
    )

    experiment.run(
        alg=alg,
        env=env,
        logger=logger,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
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