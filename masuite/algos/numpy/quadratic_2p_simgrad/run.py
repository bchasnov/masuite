import argparse
import masuite
from masuite.baselines import experiment
from masuite.baselines.numpy.constant_agent import ConstantAgent

parser = argparse.ArgumentParser()
parser.add_argument('--masuite-id', default='quadr2p', type=str,
    help='Global flag used to control which environment is loaded')
parser.add_argument('--save_path', default='tmp/masuite', type=str,
    help='where to save masuite results')
parser.add_argument('--logging_mode', default='csv', type=str,
    choices=['csv', 'sqlite', 'terminal'], help='how to log masuite results')
parser.add_argument('--overwrite', default = False, type=bool,
    help='overwrite csv logging file if found')
parser.add_argument('--num_episodes', default=None, type=int,
    help='overrides number of training episodes')

# algorithm
