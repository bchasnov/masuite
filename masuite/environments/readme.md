# Environments
The environments define the environment dynamics agents interact with. All environments inherit from the `masuite.environments.base.Environment` class. Each environment must have at least the following parameters:
* `mapping_seed: int` - defines the seed for any randomness in the enviornment.
* `n_players: int` - defines the number of players that interact with the environment.
* `env_dim: list(int)` - defines the shape of the observations the environment makes public to learning agents.
* `act_dim: list(int)` - defines the shape of the actions to be inputted by a single learning agent.
* `shared_state: bool` - defines whether or not a single state is shared between each learning agent, or if each agent has access to a different state.

In addition, each environment has two required public functions:
* `step(self, actions)` - executes a single timestep transition given actions (a list of all agents' chosen actions for that timestep). `step` returns the resulting state/observations, the calculated reward for each agent based on their actions and the resulting state, a boolean indicating whether the environment is "done" meaning it needs to be reset, and a dictionary containing any other info that could be interesting/useful.
* `reset(self)` - resets the environment to initial state and returns the initial state/observations.

Environments may also have a `render()` function to visualize their state, however this is not required.

## Loading an environment
Environments are specificed by an `env_id` string. The string consists of the environment name followed by an integer specifying the random `mapping_seed` in the following format:
```python
import masuite

env = masuite.load_env('cartpole/0')
```

The above returns a raw environment instance.

## Interacting with the environment
Each agent acts in the environment simultaneously and recieves an individual reward.
A two-agent environment takes in two actions and returns one or two observations (based on the value of the `shared_state` parameter) and two rewards. If `step()` returns `done = True` the environment should call its `reset()` function to work properly.

```python
for episode in range(20):
    env.reset()
    while True:
        act1, act2 = 0, 0
        obs, rews, done, info = env.step(act1, act2)

        # do something with obs and rews (observation and rewards)

        if done: break

print(rews)
>>> [0., 0.]
```


## Current Environments
### Cartpole (1 player)
Classic inverted-pendulum problem. A pole is attached to a cart that can move either left or right. The goal is to keep the pole upright by moving the cart left or right. The agent is rewarded if the goal is still upright after actions are executed in the environment.

### Cartpole2p (2 players)
An extension of the 1 player cartpole environment. This environment contains two inverted-pendulum carts with the same goal as above. The carts can be completely independent of one another (in which case each player's reward is calculated individually in the manner described above), or "coupled" meaning the reward calculation takes the other player's cart state into effect (this is specified by the `is_uncoupled` class parameter).

## Other environments 
The following multi-agent environments may be supported in the future:
- [ ] Quadratic game
- [ ] Linear quadratic game

  Linear Quadratic (LQ) game dynamics. Child class of [OpenAI's gym.Env](https://github.com/openai/gym/blob/c8a659369d98706b3c98b84b80a34a832bbdc6c0/gym/core.py#L8). The state update is defined as:
  
  ![equation](https://latex.codecogs.com/gif.latex?s_%7Bt&plus;1%7D%20%3D%20A%20%5Ccdot%20s_t%20&plus;%20%28%5Csum_%7Bi%3D1%7D%5E%7Bn%5C_players%7D%20B_i%20%5Ccdot%20a_i%29%20&plus;%20%5Comega)
  
  Player 1's reward in a two-player game is defined as:
  
  ![equation](https://latex.codecogs.com/gif.latex?r_1%20%3D%20-x%5ET%20%5Ccdot%20Q_1%20%5Ccdot%20x%20-%20a_1%5ET%20%5Ccdot%20R_%7B11%7D%20%5Ccdot%20a_1%20&plus;%20a_2%5ET%20%5Ccdot%20R_%7B12%7D%20%5Ccdot%20a_2)
  
- [ ] Finite matrix game
- [ ] Markov Soccer
- [ ] Cartpole 
- [ ] Prisoner's dilemma
- [ ] Bimatrix games
- [ ] ...
