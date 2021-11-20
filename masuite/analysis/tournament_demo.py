import torch
import masuite
import numpy as np
import matplotlib.pyplot as plt
import glob


def play(agent_a, agent_b, p1, p2):
    history = []
    obs = env.reset()
    done = False
    while not done:
        act1 = agent_a[p1].select_action(obs)
        act2 = agent_b[p2].select_action(obs)
        obs, rews, done, info = env.step((act1, act2))
        history.append([obs,rews,done,info])

    return rews

scores = [] 
env = masuite.load_from_id('smallsoccer/0')
simple = []
stack = []
for path in glob.iglob("../tmp/smallsoccer0-gridsearch/*.pt"):
    if "Simple" in path:
        simple.append(path)
    else:
        stack.append(path)

num = int(1e2)
tournaments1 = []
tournaments2 = []
tournaments3 = []
tournaments4 = []
labels = []

for st in stack:
    agents_a = torch.load(st)
    for si in simple:
        agents_b = torch.load(si)
        tournament1 = []
        tournament2 = []
        tournament3 = []
        tournament4 = []
        for _ in range(num):
            outcome1 = play(agents_a, agents_a, 0, 1)
            tournament1.append(outcome1)

            outcome2 = play(agents_b, agents_b, 0, 1)
            tournament2.append(outcome2)

            outcome3 = play(agents_a, agents_b, 0, 1)
            tournament3.append(outcome3)

            outcome4 = play(agents_a, agents_b, 1, 0)
            tournament4.append(outcome4)

        tournaments1.append(np.array(tournament1)[:, 0])
        tournaments2.append(np.array(tournament2)[:, 0])
        tournaments3.append(np.array(tournament3)[:, 0])
        tournaments4.append(np.array(tournament4)[:, 0])
        labels.append(st + " " + si)

sorted1 = []
for i in tournaments1:
    sorted1.append(sum(i))
sorted1.sort()

sorted2 = []
for i in tournaments2:
    sorted2.append(sum(i))
sorted2.sort()

sorted3 = []
for i in tournaments3:
    sorted3.append(sum(i))
sorted3.sort()

sorted4 = []
for i in tournaments4:
    sorted4.append(sum(i))
sorted4.sort()

plt.plot(sorted1, label="stack vs stack")
plt.plot(sorted2, label="sim vs sim")
plt.plot(sorted3, label="stack leader vs sim p2")
plt.plot(sorted4, label="stack follower vs sim p1")
plt.legend()
plt.show()
