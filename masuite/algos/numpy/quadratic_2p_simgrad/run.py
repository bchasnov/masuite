import argparse
import numpy as np
import masuite
from masuite.baselines import experiment
from masuite.baselines.numpy.constant_agent.agent import ConstantAgent
from masuite.environments.quadratic_2_player import QuadraticTwoPlayer
from masuite.algos.numpy.quadratic_2p_simgrad.quadratic_2p_simgrad import QuadraticTwoPlayerSimgrad

parser = argparse.ArgumentParser()
parser.add_argument('--masuite-id', default='quad2p', type=str,
    help='global flag used to control which environment is loaded')
parser.add_argument('--save_path', default='tmp/masuite', type=str,
    help='where to save masuite results')
parser.add_argument('--logging_mode', default='csv', type=str,
    choices=['csv', 'sqlite', 'terminal'], help='how to log masuite results')
parser.add_argument('--overwrite', default = False, type=bool,
    help='overwrite csv logging file if found')
parser.add_argument('--num_episodes', default=None, type=int,
    help='overrides number of training episodes')

# algorithm
parser.add_argument('--seed', default=0, type=int,
    help='seed for random number generation')
parser.add_argument('--lr1', default=1e-2, type=int,
    help='learning rates for agents')
parser.add_argument('--lr2', default=1e-2, type=int,
    help='learning rates for agents')

args = parser.parse_args()

def run(masuite_id: str):
    """
    Runs quadratic two-player simgrad with constant agents on a single
    masuite environment, logging to csv.
    """
    #TODO: define load_and_record()
    env = QuadraticTwoPlayer(
        A=np.array([0.7]),
        B=np.array([0.3]),
        C=np.array([0.7]),
        D=np.array([0.3])
    )

    env_dim, act_dim = 0, 1
    agents = [ConstantAgent(env_dim=env_dim, act_dim=act_dim) for _ in range(2)]
    alg = QuadraticTwoPlayerSimgrad(env=env,
        lrs=[args.lr1, args.lr2],
        agents=agents
    )

    experiment.run(
        alg=alg,
        agents=agents,
        env=env,
        num_episodes=50
    )

    return masuite_id

if __name__ == '__main__':
    run('test')