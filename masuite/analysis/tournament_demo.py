import torch
import masuite
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse 
import os
import pickle

def play(agent_a, agent_b):
    #history = []
    obs = env.reset()
    done = False
    ret = np.zeros(2)
    while not done:
        act1 = agent_a[0].select_action(obs)
        act2 = agent_b[1].select_action(obs)
        obs, rews, done, info = env.step((act1, act2))
        ret += rews
        #history.append([obs,rews,done,info])

    return ret

parser = argparse.ArgumentParser()
parser.add_argument('dir', default='../tmp/smallsoccer0-gridsearch/')
parser.add_argument('out', default='out.pkl')
parser.add_argument('--env', default='smallsoccer/0')
parser.add_argument('--num', default=int(1e1))

args = parser.parse_args()
env = masuite.load_from_id(args.env)

simple = []
stack = []
for path in glob.iglob(os.path.join(args.dir, '*.pt')):
    if "Simple" in path:
        simple.append(path)
    else:
        stack.append(path)

def tournament(agents_a, agents_b):
    tournaments = dict()
    for s_a in agents_a:
        agent_a = torch.load(s_a)
        a = os.path.basename(s_a)
        tournaments[a] = dict()
        try:
            for s_b in agents_b:
                agent_b = torch.load(s_b)
                b = os.path.basename(s_b)
                history = []
                print(a)
                print('vs')
                print(b)
                print()
                for _ in range(args.num):
                    outcome = play(agent_a, agent_b)
                    history.append(outcome)
                tournaments[a][b] = history
        except:
            print('Error!')
    return tournaments

tournaments1 = tournament(simple, simple)
tournaments2 = tournament(simple, stack)
tournaments3 = tournament(stack, simple)
tournaments4 = tournament(stack, stack)

with open(args.out, 'wb') as f:
    pickle.dump((tournaments1, tournaments2, tournaments3, tournaments4), f)
