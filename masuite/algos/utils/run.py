import argparse

# masuite imports
import masuite
from masuite.algos import experiment


parser = argparse.ArgumentParser()
# masuite logging and env params
parser.add_argument('--masuite-id', default=None, type=str,
    help='global flag used to control which environment is loaded')
parser.add_argument('--save-path', default='tmp/masuite', type=str,
    help='where to save masuite results')
parser.add_argument('--logging-mode', default='csv', type=str,
    choices=['csv', 'terminal'], help='how to log masuite results')
parser.add_argument('--overwrite', default = False, type=bool,
    help='overwrite csv logging file if found')
parser.add_argument('--log-freq', default=10, type=int,
    help='frequency at which to log env info')
parser.add_argument('--log-checkpoints', default=False, type=bool,
    help='whether or not to checkpoint agent parameters')
parser.add_argument('--checkpoint-freq', default=5, type=int,
    help='frequency (in epochs) at which to log agent checkpoints')
parser.add_argument('--log-params', default=False, type=bool,
    help='whether or not to include experiment params in log filename')
parser.add_argument('--seed', default=True, type=bool,
    help='whether or not to seed the environment')
parser.add_argument('--render', default=False, type=bool,
    help='whether or not to call the environment render function')

# algorithm-specific params
parser.add_argument('--num-epochs', default=50, type=int,
    help='number of training epochs')
parser.add_argument('--batch-size', default=5000, type=int,
    help='maximum training batch size per epoch')
parser.add_argument('--lr', default=1e-2, type=float,
    help='learning rate for agents')

args, _ = parser.parse_known_args()


def run_discrete_pg_experiment(masuite_id: str, AgentClass, AlgClass, log_to_terminal: bool=True):
    masuite_id = args.masuite_id if args.masuite_id is not None else masuite_id
    env = masuite.load_from_id(masuite_id)
    if args.seed:
        env.seed()
    
    n_players = env.n_players
    env_dim = env.env_dim
    n_acts = env.n_acts
    shared_state = env.shared_state #TODO: deprecate this

    if args.log_params:
        params = dict(
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
    else:
        params=None

    agents = [AgentClass(env_dim=env_dim, n_acts=n_acts, lr=args.lr)
        for _ in range(n_players)]

    alg = AlgClass(
        agents=agents,
        shared_state=shared_state,
        n_players=n_players,
        batch_size=args.batch_size
    )

    logger = masuite.init_logging(
        filename=f"{AlgClass.__name__}-{masuite_id}",
        n_players=n_players,
        mode=args.logging_mode,
        save_path=args.save_path,
        overwrite=args.overwrite,
        log_freq=args.log_freq,
        log_checkpoints=args.log_checkpoints,
        checkpoint_freq=args.checkpoint_freq,
        params=params
    )
    print(f'Running experiement: masuite={masuite_id}, lr={args.lr}, epochs={args.num_epochs}, batch_size={args.batch_size}')

    experiment.run(
        alg=alg,
        env=env,
        logger=logger,
        num_epochs=args.num_epochs,
        render=args.render,
        log_to_terminal=log_to_terminal
    )

    run_info = dict(
        log_save_path=logger.logger.log_save_path
    )
    return run_info
