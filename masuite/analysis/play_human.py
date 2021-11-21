import torch
import masuite
import numpy as np
import matplotlib.pyplot as plt
import sys



checkpoint_file = '../../smallsoccer0-gridsearch/SimplePG-smallsoccer-0_epochs100_batch_size1000_lr0.001_hidden_sizes[128, 64, 32]_-checkpoint.pt'

env = masuite.load_from_id('smallsoccer/0')
agents_a = torch.load(checkpoint_file)
agents_b = torch.load(checkpoint_file)


history = []
obs = env.reset()
done = False
while not done:
    act1 = agents_a[0].select_action(obs)
    act2 = agents_b[1].select_action(obs)
    obs, rews, done, info = env.step((act1, act2))
    history.append([obs,rews,done,info])

fig, ax = plt.subplots(figsize=(4,2))

pos = [(0,0), (1,0), (2,0), (3,0),
       (0,1), (1,1), (2,1), (3,1)]

COLOR1 = '#6DA195'
COLOR2 = '#CE7D85'

ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0.5,color='k')
ax.axhline(-0.5,color='k')
ax.axhline(1.5,color='k')
ax.axvline(-0.5,color='k')
ax.axvline(0.5,color='k')
ax.axvline(1.5,color='k')
ax.axvline(2.5,color='k')
ax.axvline(3.5,color='k')
ax.set(xlim=[-0.51,3.52],ylim=[-0.51,1.51],xticks=[],yticks=[])

pos1 = 0
pos2 = 1

p1,=ax.plot(*pos[pos1],'s',ms=40,color=COLOR1)
#p1_text=ax.text(*pos[pos1],'A',fontsize=24,horizontalalignment='center',verticalalignment='center')

p2,=ax.plot(*pos[pos2],'s',ms=40,color=COLOR2)
#p2_text=ax.text(*pos[pos2],'B',fontsize=24,horizontalalignment='center',verticalalignment='center')

ax.plot(*pos[1],'o',ms=35,markerfacecolor='none',markeredgecolor='black')

def on_press(event):
    global pos1
    prev_pos = pos1
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'up':
        pos1 += 4
        fig.canvas.draw()
    elif event.key == 'down':
        pos1 -= 4
        fig.canvas.draw()
    elif event.key == 'left':
        pos1 -= 1
        fig.canvas.draw()
    elif event.key == 'right':
        pos1 += 1
        fig.canvas.draw()
    if pos1 >= 8 or pos1 < 0:
        pos1 = prev_pos
    print(pos1)
    p1.set_data(*pos[pos1])
    #p2_text.set_data(*pos[pos1])
fig.canvas.mpl_connect('key_press_event', on_press)

plt.show()
